"""Small, shared checks for endpoint-dependent supervision."""

import argparse
import math

import pandas as pd

ENDPOINT_POLICY_VERSION = "selective_reception_v1"


def sample_is_eligible(value=True) -> bool:
    # Missing provenance in legacy artifacts is not evidence for exclusion.
    return pd.isna(value) or str(value).lower() not in {"false", "0"}


def valid_interval(tracking, start, end, *, allow_equal=False, period=None) -> bool:
    if pd.isna(start) or pd.isna(end):
        return False
    if end < start or (end == start and not allow_equal):
        return False
    if start not in tracking.index or end not in tracking.index:
        return False
    if "period_id" in tracking.columns:
        start_period = tracking.at[start, "period_id"]
        end_period = tracking.at[end, "period_id"]
        if pd.isna(start_period) or pd.isna(end_period) or start_period != end_period:
            return False
        if period is not None and (pd.isna(period) or start_period != period):
            return False
    return True


def nonnegative_duration(value) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("pass duration must be finite and nonnegative")
    return result
