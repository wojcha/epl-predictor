"""Exploratory analysis for the eda-target milestone.

Loads the prepared 2000+ match table, prints class balance and era-wise
missingness, and saves overview plots under reports/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Resolve paths before importing sibling modules (works when run as python src/eda.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import TARGET_COLUMN, load_prepared

# In-match / post-match fields present from 2000/01 onward in DataHub CSVs.
# Used only to measure missingness by season—not as modeling features.
_STATS_AND_META_COLUMNS = [
    "Referee",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]


def _ensure_prepared() -> pd.DataFrame:
    """Return the merged, filtered, chronologically sorted match table."""
    return load_prepared()


def ftr_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Global counts and proportions for the 3-way outcome label (H/D/A)."""
    counts = df[TARGET_COLUMN].value_counts().rename("counts")
    props = df[TARGET_COLUMN].value_counts(normalize=True).rename("proportion")
    out = pd.concat([counts, props], axis=1)
    # Keep a stable row order when all three classes are present.
    return out.loc[["H", "D", "A"]] if set(out.index) == {"H", "D", "A"} else out


def missingness_pct_by_season(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Per-season share of rows with NaN in each selected column (0–100 %)."""
    cols = [c for c in (columns or _STATS_AND_META_COLUMNS) if c in df.columns]
    rows: list[dict[str, float | int]] = []
    for season, season_matches in df.groupby("season_start_year", sort=True):
        row: dict[str, float | int] = {"season_start_year": int(season)}
        for col in cols:
            row[col] = float(season_matches[col].isna().mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("season_start_year")


def ftr_rate_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Each row is a season; columns H/D/A are outcome shares (sum to 1)."""
    counts = df.groupby("season_start_year", sort=True)[TARGET_COLUMN].value_counts(normalize=True)
    wide = counts.unstack(fill_value=0.0)
    # Rare seasons might lack a class; pad so plots always have three series.
    for label in ("H", "D", "A"):
        if label not in wide.columns:
            wide[label] = 0.0
    return wide[["H", "D", "A"]]


def goals_summary_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Match counts and mean goals per season (sanity check for data quality)."""
    working = df.copy()
    working["total_goals"] = working["FTHG"] + working["FTAG"]
    return working.groupby("season_start_year", sort=True).agg(
        matches=("FTHG", "size"),
        mean_total_goals=("total_goals", "mean"),
        mean_fthg=("FTHG", "mean"),
        mean_ftag=("FTAG", "mean"),
    )


def print_eda_tables(df: pd.DataFrame) -> None:
    """Print tabular EDA to stdout (no files written)."""
    print("Rows:", len(df))
    print("Date range:", df["Date"].min(), "->", df["Date"].max())
    print(f"\nLabel column: {TARGET_COLUMN} (H=home, D=draw, A=away)\n")

    print("=== Class balance (global) ===")
    print(ftr_class_balance(df).to_string(), end="\n\n")

    print("=== Missingness (% of rows) by season (selected columns) ===")
    miss = missingness_pct_by_season(df)
    # Only print columns that ever have gaps (keeps output readable).
    any_missing = miss.columns[miss.max(axis=0) > 0]
    if len(any_missing):
        print(miss[list(any_missing)].round(2).to_string())
    else:
        print("(No missing values in selected columns)")
    print()

    print("=== FTR rates by season (last 10 seasons) ===")
    print(ftr_rate_by_season(df).tail(10).round(4).to_string(), end="\n\n")

    print("=== Goals summary by season (last 10 seasons) ===")
    print(goals_summary_by_season(df).tail(10).round(3).to_string(), end="\n\n")


def plot_eda(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Save a three-panel figure: outcome mix, goals trend, missingness by era."""
    output_dir = output_dir or (_REPO_ROOT / "reports" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    rates = ftr_rate_by_season(df)
    goals = goals_summary_by_season(df)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    rates.plot(ax=axes[0], marker="o", ms=3, lw=1)
    axes[0].set_ylabel("Share of matches")
    axes[0].set_title(f"{TARGET_COLUMN} distribution by season")
    axes[0].legend(title="Outcome")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    goals["mean_total_goals"].plot(ax=axes[1], color="tab:blue", marker="o", ms=3)
    axes[1].set_ylabel("Mean goals")
    axes[1].set_title("Mean full-time total goals (FTHG + FTAG) by season")
    axes[1].grid(True, alpha=0.3)

    miss = missingness_pct_by_season(df)
    # Plot the eight columns with the highest peak missing rate across seasons.
    miss_max = miss.max(axis=0).sort_values(ascending=False).head(8)
    miss[miss_max.index].plot(ax=axes[2], marker="o", ms=3, lw=1)
    axes[2].set_ylabel("% of rows missing")
    axes[2].set_title("Missingness by season (top 8 columns by peak % missing)")
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "01_eda_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    prepared = _ensure_prepared()
    print_eda_tables(prepared)
    plot_eda(prepared)


if __name__ == "__main__":
    main()
