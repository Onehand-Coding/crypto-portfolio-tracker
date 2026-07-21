"""Convert pandas and datetime values into JSON-safe primitives.

The core returns DataFrames and Timestamps throughout. This module is the
single boundary where those types are converted, so no route or schema has
to know about pandas.
"""

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Recursively convert a value into something json.dumps accepts."""
    if isinstance(value, pd.DataFrame):
        return df_to_records(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return None if math.isnan(as_float) else as_float
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT or value is None:
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [jsonable(v) for v in value]
    return value


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe dicts. Empty frame -> []."""
    if df is None or df.empty:
        return []
    return [
        {str(col): jsonable(val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]
