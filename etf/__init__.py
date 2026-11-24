"""US spot bitcoin ETF holdings, flows and quarter-end cost, published with each release."""
import io
import json
from pathlib import Path
import warnings

import pandas as pd

from previous_release import previous_release_files

from .collect import CARRIED_FILES, ETF_HEALTH_FILE, TABLE_FILES, collect

ETF_FILES = TABLE_FILES + CARRIED_FILES


def get_etf_release(report_date) -> tuple[dict[str, pd.DataFrame], dict | None]:
    """Return `(files, health)` for this release. Never stops the release."""
    try:
        previous = {name: pd.read_csv(io.BytesIO(content))
                    for name, content in previous_release_files(ETF_FILES).items()}
    except Exception as error:
        warnings.warn(f"Could not read the last release's ETF files: {error}", RuntimeWarning, stacklevel=2)
        previous = {}
    try:
        files, log = collect(report_date, previous)
        return files, log.get("health")
    except Exception as error:
        if all(name in previous for name in ETF_FILES):
            warnings.warn(f"ETF collection failed ({error}); republishing the last release's ETF files",
                          RuntimeWarning, stacklevel=2)
            return previous, None
        warnings.warn(f"ETF collection failed ({error}); this release has no ETF files",
                      RuntimeWarning, stacklevel=2)
        return {}, None


def write_etf_health(health: dict | None, output_dir: str | Path = "csv") -> Path | None:
    """Write structured per-fund ETF scraper health beside the release CSVs."""
    if not health:
        return None
    target = Path(output_dir) / ETF_HEALTH_FILE
    target.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def get_etf_files(report_date) -> dict[str, pd.DataFrame]:
    """This release's ETF files. If collection fails outright, the last release's files are
    republished; with none to fall back on, the ETF files are left out. Never stops the release."""
    files, _ = get_etf_release(report_date)
    return files
