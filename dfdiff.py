import functools
import itertools
import logging
import os
import pathlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Tuple

import dvc.api
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from kedro.extras.datasets.pandas.parquet_dataset import ParquetDataSet

logger = logging.getLogger(__name__)


def resolve_filepath(dataset: str, catalog, force_env: str = None) -> str:
    kedro_dataset = getattr(catalog.datasets, dataset, None)
    if not kedro_dataset:
        raise ValueError(f"Dataset `{dataset}` not in catalog.")

    if not isinstance(kedro_dataset, ParquetDataSet):
        raise ValueError(f"Unsupported dataset type: {type(kedro_dataset)}")

    filepath = str(kedro_dataset._filepath.relative_to(os.environ["PWD"]))

    if force_env:
        filepath = filepath.replace("/crm/", f"/{force_env}/").replace(
            "/aaa/", f"/{force_env}/"
        )

    return filepath


@functools.lru_cache(maxsize=None)
def load_from_rev(
    filepath: pathlib.Path,
    rev: str = "develop",
):
    with dvc.api.open(
        filepath,
        rev=rev,
        mode="rb",
    ) as handle:
        result = pd.read_parquet(handle)

    return result


def get_git_revision_hash(revision) -> str:
    return (
        subprocess.check_output(["git", "rev-parse", revision]).decode("ascii").strip()
    )


@functools.lru_cache()
def _get_dvc_url(path: os.PathLike, rev: str) -> str:
    return dvc.api.get_url(path, repo=None, rev=rev, remote=None)


def uniquely_identifying(columns, df) -> bool:
    columns = list(columns)
    return len(df[columns].drop_duplicates()) == len(df)


def has_nondefault_index(df) -> bool:
    return df.index.names != [None]


def has_unique_index(df) -> bool:
    return not df.index.duplicated().any()


def find_uniquely_identifying_columns(
    df, max_cols: int = 5
) -> "Optional[List[Tuple[str, ...]]]":
    if has_nondefault_index(df):
        if has_unique_index(df):
            return "INDEX"
        df = df.reset_index()

    if not uniquely_identifying(df.columns, df):
        return None

    for n_columns in range(1, max_cols + 1):
        logging.debug(f"Trying combinations of {n_columns} columns")
        found_combinations = set()

        for column_combination in itertools.combinations(df.columns, n_columns):
            if uniquely_identifying(column_combination, df):
                found_combinations.add(column_combination)

        if found_combinations:
            break

    return list(found_combinations)


def is_listlike_series(series: pd.Series) -> bool:
    """Check if series likely contains list-like elements."""
    if series.isna().all():
        return False
    first_notna = series.dropna().iloc[0]
    return isinstance(first_notna, (list, np.ndarray))


def is_dict_series(series: pd.Series) -> bool:
    """Check if series likely contains list-like elements."""
    if series.isna().all():
        return False
    first_notna = series.dropna().iloc[0]
    return isinstance(first_notna, dict)


def convert_listlike_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list-like columns to tuple columns.
    This is required to allow for actions like groupby, uniques etc.
    Args:
        df: Dataframe to adjust
    Returns:
        Dataframe with list-like columns converted to tuple columns.
    """
    df = df.copy()
    for column in df:
        if is_listlike_series(df[column]):
            df[column] = df[column].apply(
                lambda value: tuple(value)
                if isinstance(value, (list, np.ndarray))
                else value
            )
            str_if_unhashable(df, column)

        elif is_dict_series(df[column]):
            df[column] = df[column].apply(
                lambda value: value.items() if isinstance(value, dict) else value
            )
            str_if_unhashable(df, column)

    return df


def mask_if_unhashable(df, column):
    try:
        df[column].drop_duplicates()
    except TypeError:
        logging.info(f"Masking column `{column}`.")
        df[column] = "MASKED"


def str_if_unhashable(df: pd.DataFrame, column: str) -> None:
    try:
        df[column].drop_duplicates()
    except TypeError:
        logging.info(f"Converting column `{column}` to string dtype.")
        df[column] = df[column].astype(str)


def ensure_hashable(series: pd.Series) -> pd.Series:
    try:
        _ = series.drop_duplicates()
        return series
    except TypeError:
        logging.warning(f"Converting series `{series.name}` to string dtype.")
        return series.astype(pd.StringDtype())


def _collapsible_df(df: pd.DataFrame, name: str, max_rows: int = 60) -> str:
    return f"""<details><summary><h4>{name} ({len(df)} records)</h4></summary>
<i>Showing up to {max_rows} rows.</i>
{df.head(max_rows).to_html()}
</details>
"""


class DfDiff:
    def __init__(
        self,
        df1,
        df2,
        *,
        index_cols=None,
        # value_cols=None,
        # meta_cols=None,
    ):
        # assert not df1.equals(df2)
        assert df1.columns.equals(df2.columns)
        assert not df1.duplicated(subset=index_cols).any()
        assert not df2.duplicated(subset=index_cols).any()
        assert df1.index.names == df2.index.names == [None]

        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.index_cols = index_cols
        self.indices = None
        self.subsets = None
        self.common_diff = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: reset index -> name
        df = df.apply(ensure_hashable)
        df = df.set_index(self.index_cols)
        return df

    def calculate(self):
        self._df1 = self._prepare(self.df1)
        self._df2 = self._prepare(self.df2)

        self.indices = self._categorise_indices(self._df1, self._df2)

        self.subsets = {
            "df1_only": self._df1.loc[self.indices["df1_only"]],
            "df2_only": self._df2.loc[self.indices["df2_only"]],
            "df1_overlap": self._df1.loc[self.indices["both"]],
            "df2_overlap": self._df2.loc[self.indices["both"]],
        }

        # TODO: rename columns - (self, other) -> (current, reference)
        self.common_diff = self._calculate_value_diff(
            self.subsets["df1_overlap"], self.subsets["df2_overlap"]
        )

    def _find_diffs(self, series1: pd.Series, series2: pd.Series):
        assert series1.index.equals(series2.index)
        assert series1.dtype == series2.dtype
        if pd.api.types.is_float_dtype(series1.dtype):
            return np.isclose(series1, series2, atol=1e-7) | (
                series1.isna() & series2.isna()
            )
        return series1 == series2

    def _calculate_value_diff(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        df1 = df1.sort_index()
        df2 = df2.sort_index()[df1.columns]

        diff_masks: Dict[str, pd.Series] = {}
        for (name, series1), (name, series2) in zip(df1.items(), df2.items()):
            diff_masks[name] = ~self._find_diffs(series1, series2)

        diff_df = (
            pd.DataFrame(diff_masks).assign(
                # cols=lambda d: d.apply(tuple, axis=1),
                cols=lambda d: d.apply(
                    lambda row: tuple(d.columns[row]), axis=1
                ),  # TODO: slow
                # n_cols=lambda d: d["cols"].str.len(),
                # n_records=lambda d: d.groupby("cols")["cols"].transform("size"),
            )
            # .loc[lambda d: d["n_cols"] > 0]
        )

        subset_comparisons = {}
        for cols, diff_subset in diff_df.groupby("cols"):
            if (n_cols := len(cols)) == 0:
                continue

            subset_index = diff_subset.index
            subset_comparison = df1.loc[subset_index, list(cols)].compare(
                df2.loc[subset_index, list(cols)]
            )
            subset_comparison.columns = subset_comparison.columns.set_levels(
                ["current", "reference"], level=1
            )

            extra_metadata = df1[list(set(df1) - set(cols))]
            extra_metadata.columns = pd.MultiIndex.from_product(
                [extra_metadata, ["(no diff)"]]
            )

            subset_comparison = subset_comparison.join(extra_metadata)
            subset_comparisons[cols] = subset_comparison

            print(cols)
            display(subset_comparison)

        # display(subset_comparisons)
        # display(pd.concat(subset_comparisons.values()))
        # name_groups = _group_series(diff_masks)

        # for name_group in name_groups:
        #     diff_mask = diff_masks[name_group[0]]
        #     if not diff_mask.any():
        #         continue
        #     xx = df1.loc[diff_mask, name_group].compare(
        #         df2.loc[diff_mask, name_group],
        #     )
        #     xx.columns = xx.columns.set_levels(["current", "reference"], level=1)
        #     print(name_group)
        #     display(xx)

        df_comp = df1.compare(df2)
        df_comp.columns = df_comp.columns.set_levels(["current", "reference"], level=1)
        return df_comp

    def _categorise_indices(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> Dict[str, pd.Index]:
        # TODO: why different? for tuples?
        # df_merged = pd.merge(df1, df2, on=self.index_cols, how="outer", indicator=True)

        df_merged = pd.merge(
            df1, df2, left_index=True, right_index=True, how="outer", indicator=True
        )
        # print(
        #     f"{len(df1)=}, {len(df2)=}, \n{df_merged._merge.value_counts(dropna=False)}"
        # )

        left_only_idx = df_merged.loc[lambda d: d["_merge"] == "left_only"].index
        right_only_idx = df_merged.loc[lambda d: d["_merge"] == "right_only"].index
        both_idx = df_merged.loc[lambda d: d["_merge"] == "both"].index

        assert set(left_only_idx).union(both_idx) == set(df1.index)
        assert set(right_only_idx).union(both_idx) == set(df2.index)

        df1.loc[set(left_only_idx).union(both_idx)]
        df2.loc[set(right_only_idx).union(both_idx)]

        return {
            "df1_only": left_only_idx,
            "df2_only": right_only_idx,
            "both": both_idx,
        }

    def is_calculated(self) -> bool:
        # TODO
        return all(
            (
                # self.indicator_overlap is not None,
                # self.df1_only is not None,
                # self.df2_only is not None,
                self.common_diff
                is not None,
            )
        )

    def display(self) -> None:
        if not self.is_calculated():
            raise ValueError("Diff not calculated yet.")

        display(
            HTML(_collapsible_df(self.subsets["df1_overlap"], "🟰 Matching")),
            # TODO: improve display of Changed - isclose; per column / cluster;
            # self/other
            HTML(_collapsible_df(self.common_diff, "🔀 Changed")),
            HTML(_collapsible_df(self.subsets["df1_only"], "➕ Added")),
            HTML(_collapsible_df(self.subsets["df2_only"], "➖ Dropped")),
        )

    def __call__(self):
        self.calculate()
        self.display()


def _hash_series(series):
    return series.to_numpy().tobytes()


def _group_series(series_map: Mapping[str, pd.Series]) -> List[List[str]]:
    col_groups = defaultdict(list)
    for name, col in series_map.items():
        hsh = _hash_series(col)
        col_groups[hsh].append(name)

    return list(col_groups.values())


def validate(
    catalog,
    dataset: str,
    rev_reference: str,
    rev_current: str = "HEAD",
) -> Optional[DfDiff]:

    filepath = resolve_filepath(dataset, catalog)
    logger.info(f"Resolved filepath: {filepath}")

    hash_current = get_git_revision_hash(rev_current)
    hash_reference = get_git_revision_hash(rev_reference)
    path_current = _get_dvc_url(filepath, rev=hash_current)
    path_reference = _get_dvc_url(filepath, rev=hash_reference)

    if path_current == path_reference:
        logger.info("Identical files")
        return None

    df_current = load_from_rev(filepath, rev=hash_current)
    df_reference = load_from_rev(filepath, rev=hash_reference)

    df_current = df_current.pipe(convert_listlike_columns)
    df_reference = df_reference.pipe(convert_listlike_columns)

    if df_current.duplicated().any():
        raise ValueError("Current dataframe is duplicated")
    if df_reference.duplicated().any():
        raise ValueError("Reference dataframe is duplicated")

    index_cols_opts = find_uniquely_identifying_columns(
        df_current.select_dtypes(exclude=float)
    )
    assert isinstance(index_cols_opts, list), "Indices not supported as identifiers yet"

    index_cols = list(index_cols_opts[0])
    logger.info(f"Resolved index_cols: {index_cols}")
    dfd = DfDiff(df_current, df_reference, index_cols=index_cols.copy())
    try:
        dfd.calculate()

        display(HTML(f"<h3>{dataset}</h3>"))
        dfd.display()
    except Exception as e:
        logger.warning(e)

    return dfd
