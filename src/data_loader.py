
from pathlib import Path

import pandas as pd


SUPPORTED_FORMATS = {".csv"}


def load_feedback_data(source, file_name=None):
    """
    Load consumer feedback data from a CSV file.

    Parameters
    ----------
    source : str, Path or file-like object
        File path or uploaded file.
    file_name : str, optional
        Original filename, used to identify the file format.

    Returns
    -------
    pd.DataFrame
        Loaded feedback dataset.
    """

    if file_name:
        suffix = Path(file_name).suffix.lower()
    elif isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lower()
    else:
        suffix = ".csv"

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Please upload a CSV file."
        )

    encodings = ["utf-8-sig", "utf-8", "gb18030"]

    last_error = None

    for encoding in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)

            df = pd.read_csv(source, encoding=encoding)

            if df.empty:
                raise ValueError("The uploaded dataset is empty.")

            df.columns = [
                str(column).strip()
                for column in df.columns
            ]

            if len(df.columns) != len(set(df.columns)):
                raise ValueError(
                    "Duplicate column names detected. "
                    "Please check the dataset."
                )

            return df

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"Unable to decode the CSV file: {last_error}"
    )


def get_data_overview(df):
    """Generate a basic overview of the feedback dataset."""

    if df is None or df.empty:
        raise ValueError("No valid data available.")

    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_records": int(df.duplicated().sum()),
        "columns": df.columns.tolist(),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / 1024**2, 2
        ),
    }
