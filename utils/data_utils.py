import pandas as pd
import numpy as np
import json
from io import StringIO
from config.settings import SAMPLE_SIZE


def load_csv(file_obj, sample=False):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_obj: File-like object containing CSV data
        sample (bool): Whether to sample large files

    Returns:
        pandas.DataFrame: Loaded DataFrame
    """
    try:
        # Read the first few rows to determine file size
        preview = pd.read_csv(file_obj, nrows=5)
        file_obj.seek(0)  # Reset file pointer

        # For large files, use sampling if requested
        if sample:
            df = pd.read_csv(
                file_obj,
                skiprows=lambda i: i > 0
                and np.random.random() > SAMPLE_SIZE / file_obj.tell(),
            )
        else:
            df = pd.read_csv(file_obj)

        return df
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")


def get_dataframe_info(df):
    """
    Generate a comprehensive profile of the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to profile

    Returns:
        dict: Profile information
    """
    # Basic info
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "unique_values": {col: int(df[col].nunique()) for col in df.columns},
    }

    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        info["numeric_stats"] = {}
        for col in numeric_cols:
            info["numeric_stats"][col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median())
                if not pd.isna(df[col].median())
                else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
            }

    # Identify potential date columns
    date_candidates = []
    for col in df.columns:
        if df[col].dtype == "object":
            # Try parsing as date
            try:
                pd.to_datetime(df[col], errors="raise")
                date_candidates.append(col)
            except:
                pass

    info["date_candidates"] = date_candidates

    # Sample values for categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        info["categorical_samples"] = {}
        for col in categorical_cols:
            # Get up to 10 unique values as samples
            samples = df[col].dropna().unique()[:10].tolist()
            info["categorical_samples"][col] = [str(s) for s in samples]

    # Add correlation for numeric columns if there are multiple
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(2)
        # Convert to dictionary format
        corr_dict = {}
        for col1 in numeric_cols:
            corr_dict[col1] = {}
            for col2 in numeric_cols:
                if col1 != col2:  # Skip self-correlations
                    corr_value = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_value):
                        corr_dict[col1][col2] = float(corr_value)

        info["correlations"] = corr_dict

    return info


def dataframe_to_str(df, limit=10):
    """
    Convert DataFrame to a string representation.

    Args:
        df (pandas.DataFrame): DataFrame to stringify
        limit (int): Max rows to include

    Returns:
        str: String representation of DataFrame
    """
    buffer = StringIO()

    if len(df) <= limit * 2:
        df.to_string(buffer)
    else:
        # Show head and tail if large
        pd.concat([df.head(limit), df.tail(limit)]).to_string(buffer)

    return buffer.getvalue()


def apply_preprocessing(df, steps):
    """
    Apply preprocessing steps to DataFrame.

    Args:
        df (pandas.DataFrame): Original DataFrame
        steps (list): List of preprocessing step dictionaries

    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    result_df = df.copy()

    for step in steps:
        step_type = step.get("type")

        if step_type == "missing_values":
            method = step.get("method")
            columns = step.get("columns", [])

            if method == "drop_rows":
                result_df = result_df.dropna(subset=columns)

            elif method == "imputation":
                strategy = step.get("parameters", {}).get("strategy", "mean")

                for col in columns:
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(
                        result_df[col]
                    ):
                        result_df[col] = result_df[col].fillna(result_df[col].mean())
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(
                        result_df[col]
                    ):
                        result_df[col] = result_df[col].fillna(result_df[col].median())
                    elif strategy == "mode":
                        result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                    elif strategy == "constant":
                        fill_value = step.get("parameters", {}).get("value")
                        result_df[col] = result_df[col].fillna(fill_value)

        elif step_type == "date_conversion":
            columns = step.get("columns", [])

            for col in columns:
                try:
                    # Make sure to store as datetime64 dtype
                    result_df[col] = pd.to_datetime(result_df[col], errors="coerce")
                    print(
                        f"Converted {col} to datetime. New dtype: {result_df[col].dtype}"
                    )
                except Exception as e:
                    print(f"Error converting {col} to datetime: {str(e)}")

        elif step_type == "standardization":
            columns = step.get("columns", [])

            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    result_df[col] = (result_df[col] - mean) / std

        elif step_type == "normalization":
            columns = step.get("columns", [])

            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    result_df[col] = (result_df[col] - min_val) / (max_val - min_val)

        elif step_type == "categorical_encoding":
            method = step.get("method")
            columns = step.get("columns", [])

            for col in columns:
                if method == "one_hot":
                    # Get dummies and join with original dataframe
                    dummies = pd.get_dummies(result_df[col], prefix=col)
                    result_df = pd.concat([result_df, dummies], axis=1)
                    result_df = result_df.drop(col, axis=1)

                elif method == "label":
                    result_df[f"{col}_encoded"] = (
                        result_df[col].astype("category").cat.codes
                    )

    return result_df
