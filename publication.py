"""Number precision shared by every published file."""

import numpy as np
import pandas as pd

# Ten significant digits. Values from 10 billion to 1e15 (market caps, volumes) keep whole
# units instead of switching to scientific notation; beyond that (hash rate) they must stay
# in scientific notation to read back as numbers.
SIGNIFICANT_DIGITS = 10
WHOLE_NUMBER_RANGE = (1e10, 1e15)


def format_float(value: float) -> str:
    """The published text for one float; blank when it is not finite."""
    if not np.isfinite(value):
        return ""
    if WHOLE_NUMBER_RANGE[0] <= abs(value) < WHOLE_NUMBER_RANGE[1]:
        return f"{value:.0f}"
    return f"{value:.{SIGNIFICANT_DIGITS}g}"


def round_for_publication(frame: pd.DataFrame) -> pd.DataFrame:
    """Round every float column to its published value.

    Tables built from the rounded frame then agree exactly with the master file readers see.
    """
    frame = frame.copy()
    for column in frame.select_dtypes("float").columns:
        frame[column] = [
            float(format_float(value)) if np.isfinite(value) else value
            for value in frame[column]
        ]
    return frame
