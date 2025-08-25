"""Utility for loading CSV data with caching.

Ensures each CSV file is read only once per process. Subsequent
calls return a copy of the cached DataFrame to avoid unintended
modifications across modules.
"""
from functools import lru_cache
from pathlib import Path
from typing import Tuple
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def _make_key(filename: str, kwargs: dict) -> Tuple[str, Tuple[Tuple[str, object], ...]]:
    """Create a hashable cache key from filename and keyword arguments."""
    return filename, tuple(sorted(kwargs.items()))


@lru_cache(maxsize=None)
def _read_csv_cached(filename: str, kwargs_tuple: Tuple[Tuple[str, object], ...]) -> pd.DataFrame:
    """Read a CSV file and cache the resulting DataFrame."""
    kwargs = dict(kwargs_tuple)
    return pd.read_csv(DATA_DIR / filename, **kwargs)


def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file from the backend directory with caching.

    Parameters
    ----------
    filename: str
        Name of the CSV file located within the backend directory.
    **kwargs:
        Additional keyword arguments passed to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        A copy of the cached DataFrame.
    """
    df = _read_csv_cached(filename, _make_key(filename, kwargs)[1])
    return df.copy()

