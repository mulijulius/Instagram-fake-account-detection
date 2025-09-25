import os
import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    """Load a tabular file into a DataFrame with basic format autodetection.

    Supported formats by file extension:
    - .csv, .txt  -> CSV (comma-separated by default)
    - .xlsx, .xls -> Excel
    - .json       -> JSON (records or table)
    - .parquet    -> Parquet (requires pyarrow or fastparquet)

    Parameters
    ----------
    path: str
        Absolute or relative path to the file on disk.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".csv", ".txt"):
            # Fallback to default CSV reader; for TSVs user can open explicitly
            return pd.read_csv(path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if ext == ".json":
            # Let pandas infer the JSON orientation when possible
            return pd.read_json(path)
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - surface a clear error upstream
        raise RuntimeError(f"Failed to load file '{path}': {exc}") from exc

    raise ValueError(f"Unsupported file type for '{path}'. Supported: CSV, Excel, JSON, Parquet")


def load_dataset(path: str) -> pd.DataFrame:
    """Backward-compatible dataset loader used by the modeling pipeline.

    This now delegates to `load_table` to support several common formats while
    keeping the original function name used across the app.
    """
    return load_table(path)
