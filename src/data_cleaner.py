
import pandas as pd
from collections import Counter


def clean_feedback_data(
    df,
    rating_column="rating",
    date_column="date",
    feedback_column="feedback_text",
):
    """
    Clean consumer feedback data, trace anomalies,
    and generate a data quality report.

    Returns
    -------
    cleaned_df : pd.DataFrame
    report : dict
    """

    if df is None or df.empty:
        raise ValueError("Input dataset is empty.")

    original_df = df.copy().reset_index(drop=True)
    cleaned_df = original_df.copy()

    # 1. Standardize column names
    cleaned_df.columns = [
        str(column).strip()
        for column in cleaned_df.columns
    ]
    original_df.columns = cleaned_df.columns
    
    if cleaned_df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    original_rows = len(cleaned_df)
    # Count missing values in the original dataset
    missing_before = int(
        original_df.isna().sum().sum()
    )
    # CSV row numbers: row 1 is the header
    source_rows = {
        index: index + 2
        for index in cleaned_df.index
    }

    anomaly_records = []

    def safe_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def add_anomaly(index, issue_type, original_value=None,
                    details=""):
        feedback_id = None

        if "feedback_id" in original_df.columns:
            feedback_id = safe_value(
                original_df.at[index, "feedback_id"]
            )

        anomaly_records.append({
            "source_row": source_rows[index],
            "feedback_id": feedback_id,
            "issue_type": issue_type,
            "original_value": safe_value(original_value),
            "details": details,
        })

    # 2. Remove surrounding whitespace
    for column in cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns:
        cleaned_df[column] = cleaned_df[column].map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

    # 3. Convert blank strings into missing values
    cleaned_df = cleaned_df.replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

    missing_after_standardization = int(
        cleaned_df.isna().sum().sum()
    )

    # 4. Detect and remove exact duplicate records
    duplicate_mask = cleaned_df.duplicated(
        keep="first"
    )

    duplicate_count = int(duplicate_mask.sum())

    for index in cleaned_df.index[duplicate_mask]:
        original_record = {
            column: safe_value(cleaned_df.at[index, column])
            for column in cleaned_df.columns
        }

        add_anomaly(
            index,
            "exact_duplicate",
            original_value=original_record,
            details="Exact duplicate record removed."
        )

    cleaned_df = cleaned_df.loc[
        ~duplicate_mask
    ].copy()

    # 5. Detect duplicate feedback IDs
    duplicate_feedback_ids = 0

    if "feedback_id" in cleaned_df.columns:
        duplicate_id_mask = (
            cleaned_df["feedback_id"].notna()
            & cleaned_df["feedback_id"].duplicated(
                keep="first"
            )
        )

        duplicate_feedback_ids = int(
            duplicate_id_mask.sum()
        )

        for index in cleaned_df.index[duplicate_id_mask]:
            add_anomaly(
                index,
                "duplicate_feedback_id",
                cleaned_df.at[index, "feedback_id"],
                "Feedback ID already exists in an earlier record."
            )

    # 6. Validate rating values
    invalid_ratings = 0

    if rating_column in cleaned_df.columns:
        original_rating = cleaned_df[rating_column]

        numeric_rating = pd.to_numeric(
            original_rating,
            errors="coerce",
        )

        invalid_mask = (
            numeric_rating.isna()
            & original_rating.notna()
        ) | (
            numeric_rating.notna()
            & ~numeric_rating.between(1, 5)
        )

        invalid_ratings = int(invalid_mask.sum())

        for index in cleaned_df.index[invalid_mask]:
            add_anomaly(
                index,
                "invalid_rating",
                original_rating.loc[index],
                "Rating must be a numeric value between 1 and 5."
            )

        cleaned_df[rating_column] = numeric_rating.mask(
            invalid_mask
        )

    # 7. Standardize and validate date values
    invalid_dates = 0

    if date_column in cleaned_df.columns:
        original_date = cleaned_df[date_column]

        parsed_date = pd.to_datetime(
            original_date,
            errors="coerce",
        )

        invalid_date_mask = (
            parsed_date.isna()
            & original_date.notna()
        )

        invalid_dates = int(
            invalid_date_mask.sum()
        )

        for index in cleaned_df.index[invalid_date_mask]:
            add_anomaly(
                index,
                "invalid_date",
                original_date.loc[index],
                "Date could not be parsed."
            )

        cleaned_df[date_column] = parsed_date

    # 8. Check missing feedback IDs
    missing_feedback_ids = 0

    if "feedback_id" in cleaned_df.columns:
        missing_id_mask = cleaned_df["feedback_id"].isna()
        missing_feedback_ids = int(missing_id_mask.sum())

        for index in cleaned_df.index[missing_id_mask]:
            add_anomaly(
                index,
                "missing_feedback_id",
                details="Feedback ID is missing."
            )

    # 9. Check missing feedback text
    missing_feedback = 0

    if feedback_column in cleaned_df.columns:
        missing_feedback_mask = (
            cleaned_df[feedback_column].isna()
        )

        missing_feedback = int(
            missing_feedback_mask.sum()
        )

        for index in cleaned_df.index[missing_feedback_mask]:
            add_anomaly(
                index,
                "missing_feedback",
                details="Feedback text is missing."
            )

    # 10. Generate anomaly summary
    anomaly_summary = dict(
        Counter(
            record["issue_type"]
            for record in anomaly_records
        )
    )

    # 11. Generate data quality report
    report = {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned_df),
        "duplicates_removed": duplicate_count,
        "duplicate_feedback_ids": duplicate_feedback_ids,
        "missing_values_before": missing_before,
        "missing_after_standardization": missing_after_standardization,
        "missing_values_after": int(
            cleaned_df.isna().sum().sum()
        ),
        "invalid_ratings": invalid_ratings,
        "invalid_dates": invalid_dates,
        "missing_feedback_ids": missing_feedback_ids,
        "missing_feedback": missing_feedback,
        "data_retention_rate": round(
            len(cleaned_df) / original_rows * 100, 2
        ),
        "anomaly_count": len(anomaly_records),
        "anomaly_summary": anomaly_summary,
        "anomaly_records": anomaly_records,
    }

    return cleaned_df.reset_index(drop=True), report
