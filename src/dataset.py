"""Load EPL season CSVs from disk, merge, and return a cleaned chronological table.
Supervised target for match-outcome models: column ``FTR`` (H=home, D=draw, A=away).
Do not use in-match or post-match columns as *pre-match* features (e.g. shots, cards).
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

# First season with full stats in the dataset
MIN_SEASON_START_YEAR = 2000

TARGET_COLUMN = "FTR"

# Relative to project root when callers pass None
DEFAULT_RAW_DIR = Path("data/raw")


def season_stem_to_start_year(filename_stem: str) -> int:
    """Map a ``season-YYZZ`` stem to the calendar year that starts the season (YY).

    Filenames use two-digit years: ``season-9394`` is 1993/94; ``season-0001`` is 2000/01.
    We treat YY >= 93 as 19YY and YY < 93 as 20YY so older decades sort correctly.
    """
    season_year_match = re.search(r"season-(\d{2})(\d{2})", filename_stem, re.IGNORECASE)
    if not season_year_match:
        raise ValueError(f"Invalid filename stem: {filename_stem}")
    season_start = int(season_year_match.group(1))
    return season_start + (
        1900 if season_start >= 93 else 2000
    )


def load_raw_matches(local_raw_data_directory: Path | None = None) -> pd.DataFrame:
    """Read every ``season-*.csv`` under ``data/raw``, tag rows with season metadata, concat."""
    local_raw_data_directory = local_raw_data_directory or DEFAULT_RAW_DIR
    sorted_season_csv_paths = sorted(local_raw_data_directory.glob("season-*.csv"))
    if not sorted_season_csv_paths:
        raise FileNotFoundError(f"No season CSV files found in {local_raw_data_directory}")

    season_dataframe_list: list[pd.DataFrame] = []
    for season_csv_path in sorted_season_csv_paths:
        season_file_stem = season_csv_path.stem
        season_calendar_start_year = season_stem_to_start_year(season_file_stem)
        # Football-Data CSVs use Latin-1 for accented club names
        per_season_matches_df = pd.read_csv(season_csv_path, encoding="ISO-8859-1")
        per_season_matches_df["season_start_year"] = season_calendar_start_year
        per_season_matches_df["season_key"] = season_file_stem.replace("season-", "").lower()
        season_dataframe_list.append(per_season_matches_df)

    return pd.concat(season_dataframe_list, ignore_index=True)


def prepare_matches(raw_combined_matches_df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, drop unusable rows, coerce scores, sort by kickoff then season (overlap safe)."""
    prepared_matches_df = raw_combined_matches_df.copy()
    # UK-style dates in source files (DD/MM/YYYY)
    prepared_matches_df["Date"] = pd.to_datetime(prepared_matches_df["Date"], dayfirst=True, errors="coerce")
    prepared_matches_df = prepared_matches_df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    prepared_matches_df = prepared_matches_df[prepared_matches_df["FTR"].isin(["H", "D", "A"])]
    prepared_matches_df = prepared_matches_df[prepared_matches_df["season_start_year"] >= MIN_SEASON_START_YEAR]

    for full_time_goals_column in ["FTHG", "FTAG"]:
        if full_time_goals_column in prepared_matches_df.columns:
            prepared_matches_df[full_time_goals_column] = pd.to_numeric(prepared_matches_df[full_time_goals_column], errors="coerce")
    prepared_matches_df = prepared_matches_df.dropna(subset=["FTHG", "FTAG"])
    # season_start_year breaks ties when the same calendar date appears in adjacent season files
    prepared_matches_df = prepared_matches_df.sort_values(["Date", "season_start_year"]).reset_index(drop=True)
    return prepared_matches_df


def load_prepared(local_raw_data_directory: Path | None = None) -> pd.DataFrame:
    """Convenience: raw concat + ``prepare_matches`` in one call."""
    return prepare_matches(load_raw_matches(local_raw_data_directory))


def describe_dataset(prepared_matches_df: pd.DataFrame) -> None:
    """Print quick EDA: shape, date span, class balance, missingness, recent season counts."""

    print("Rows", len(prepared_matches_df))
    print(
        "Date range", prepared_matches_df["Date"].min(), "->", prepared_matches_df["Date"].max()
    )
    print("\nTarget y = FTR (match outcome):")
    print(prepared_matches_df["FTR"].value_counts())
    print("\nFTR proportions:")
    print(prepared_matches_df["FTR"].value_counts(normalize=True).round(4))
    print("\nMissing values (% of rows), worst columns:")
    missing_fraction_by_column = prepared_matches_df.isna().mean().sort_values(ascending=False)
    missing_percent_top = (missing_fraction_by_column.head(15) * 100).round(2)
    print(missing_percent_top.to_string())
    print("\nMatches per season_start_year (last 8 seasons):")
    matches_per_season = prepared_matches_df.groupby("season_start_year").size().tail(8)
    print(matches_per_season.to_string())

