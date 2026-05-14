"""Download EPL season CSVs from DataHub using the dataset datapackage.json."""

import sys
from pathlib import Path

import requests

# JSON catalog listing every resource (each season file is a resource with a relative path).
DATAPACKAGE_URL = "https://datahub.io/core/english-premier-league/datapackage.json"
# Prefix for raw file bytes; append resource "path" (e.g. season-2425.csv).
CSV_BASE_URL = "https://datahub.io/core/english-premier-league/r/"
# Relative to the process working directory (run from project root).
LOCAL_RAW_DATA_DIR = Path("data/raw")


def fetch_datapackage() -> dict:
    """Load and parse datapackage.json (Frictionless Data package descriptor)."""
    datapackage_response = requests.get(DATAPACKAGE_URL, timeout=60)
    datapackage_response.raise_for_status()
    return datapackage_response.json()


def download_all(skip_existing: bool = True) -> None:
    """Fetch each season CSV listed in the datapackage into LOCAL_RAW_DATA_DIR."""
    LOCAL_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    datapackage = fetch_datapackage()
    for resource in datapackage.get("resources", []):
        relative_path = resource.get("path", "")
        # Skip README and other non-tabular resources.
        if not relative_path.endswith(".csv"):
            continue
        # Write flat under data/raw/ even if path ever contained directory segments.
        output_file_path = LOCAL_RAW_DATA_DIR / Path(relative_path).name
        if skip_existing and output_file_path.exists():
            continue
        url = CSV_BASE_URL + relative_path
        csv_response = requests.get(url, timeout=120)
        csv_response.raise_for_status()
        output_file_path.write_bytes(csv_response.content)


def main() -> None:
    try:
        download_all()
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    csv_count = len(list(LOCAL_RAW_DATA_DIR.glob("season-*.csv")))
    print(f"Done. Found {csv_count} season CSV files in {LOCAL_RAW_DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
