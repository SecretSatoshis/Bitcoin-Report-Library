"""US spot bitcoin ETF holdings, flows and quarter-end cost, published with each release."""
import io
import warnings

import pandas as pd

from previous_release import previous_release_files

from .collect import CARRIED_FILES, TABLE_FILES, collect

ETF_FILES = TABLE_FILES + CARRIED_FILES


def get_etf_files(report_date) -> dict[str, pd.DataFrame]:
    """This release's ETF files. If collection fails outright, the last release's files are
    republished; with none to fall back on, the ETF files are left out. Never stops the release."""
    try:
        previous = {name: pd.read_csv(io.BytesIO(content))
                    for name, content in previous_release_files(ETF_FILES).items()}
    except Exception as error:
        warnings.warn(f"Could not read the last release's ETF files: {error}", RuntimeWarning, stacklevel=2)
        previous = {}
    try:
        files, _ = collect(report_date, previous)
        return files
    except Exception as error:
        if all(name in previous for name in ETF_FILES):
            warnings.warn(f"ETF collection failed ({error}); republishing the last release's ETF files",
                          RuntimeWarning, stacklevel=2)
            return previous
        warnings.warn(f"ETF collection failed ({error}); this release has no ETF files",
                      RuntimeWarning, stacklevel=2)
        return {}
