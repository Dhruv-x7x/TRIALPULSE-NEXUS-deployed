import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Any
import math

def sanitize_for_json(data: Any) -> Any:
    """
    Recursively sanitize data for JSON serialization.
    - Replaces NaN/Inf with None
    - Converts numpy types to native Python types
    - Converts datetime/date to ISO strings
    - Handles pandas Timestamps
    """
    if isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, (np.floating, np.integer, np.number)):
        # Handle numpy specific types
        if np.isnan(data) or np.isinf(data):
            return None
        if isinstance(data, (np.floating, float)):
            return float(data)
        return int(data)
    elif isinstance(data, (pd.Timestamp, datetime)):
        return data.isoformat() if pd.notna(data) else None
    elif isinstance(data, date):
        return data.isoformat() if data else None
    elif pd.isna(data):
        return None
    else:
        return data
