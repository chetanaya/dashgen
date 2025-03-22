import pandas as pd
from utils.openai_utils import analyze_data_schema
from utils.data_utils import get_dataframe_info, dataframe_to_str


class DataProfilerAgent:
    """
    Agent responsible for generating insights about the dataset structure,
    quality, and potential analyses.
    """

    def __init__(self):
        """Initialize the DataProfilerAgent."""
        pass

    def profile_dataset(self, df):
        """
        Generate a comprehensive profile of the dataset.

        Args:
            df (pandas.DataFrame): The dataset to profile

        Returns:
            dict: A complete profile of the dataset
        """
        # Get basic DataFrame information
        df_info = get_dataframe_info(df)

        # Explicitly identify datetime columns
        datetime_columns = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_columns:
            df_info["date_candidates"] = datetime_columns

        # Convert sample of DataFrame to string for OpenAI analysis
        df_str = dataframe_to_str(df)

        # Get AI analysis of the data
        ai_analysis = analyze_data_schema(df_str, df_info)

        # Combine all profiling information
        profile = {
            "basic_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            },
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {
                "total": df.isna().sum().sum(),
                "by_column": {col: int(df[col].isna().sum()) for col in df.columns},
            },
            "unique_values": {col: int(df[col].nunique()) for col in df.columns},
            "statistics": df.describe().to_dict(),
            "ai_analysis": ai_analysis,
            "quality_issues": self._identify_quality_issues(df, df_info),
            "suggested_preprocessing": self._suggest_preprocessing(
                df, df_info, ai_analysis
            ),
            "potential_analyses": ai_analysis.get("potential_analyses", [])
            if isinstance(ai_analysis, dict)
            else [],
        }

        return profile

    def _identify_quality_issues(self, df, df_info):
        """
        Identify potential data quality issues.

        Args:
            df (pandas.DataFrame): The dataset
            df_info (dict): Data profile information

        Returns:
            list: Identified quality issues
        """
        issues = []

        # Check for missing values
        missing_cols = [
            col for col, count in df_info["missing_values"].items() if count > 0
        ]
        if missing_cols:
            issues.append(
                {
                    "type": "missing_values",
                    "description": f"Found missing values in {len(missing_cols)} column(s)",
                    "affected_columns": missing_cols,
                    "severity": "high"
                    if any(
                        df_info["missing_values"][col] / df.shape[0] > 0.2
                        for col in missing_cols
                    )
                    else "medium",
                }
            )

        # Check for duplicated rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(
                {
                    "type": "duplicates",
                    "description": f"Found {duplicate_count} duplicate row(s)",
                    "affected_rows": int(duplicate_count),
                    "severity": "medium"
                    if duplicate_count / df.shape[0] > 0.05
                    else "low",
                }
            )

        # Check for potential outliers in numeric columns
        for col in df.select_dtypes(include=["number"]).columns:
            if col in df_info.get("numeric_stats", {}):
                stats = df_info["numeric_stats"][col]

                # Use IQR to detect outliers
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)

                outlier_count = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).sum()

                if outlier_count > 0:
                    issues.append(
                        {
                            "type": "outliers",
                            "description": f"Found {outlier_count} potential outlier(s) in column '{col}'",
                            "affected_column": col,
                            "affected_rows": int(outlier_count),
                            "severity": "medium"
                            if outlier_count / df.shape[0] > 0.05
                            else "low",
                        }
                    )

        # Check for inconsistent data types
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if column might actually be numeric
                numeric_values = pd.to_numeric(df[col], errors="coerce")
                if (
                    numeric_values.notna().sum() / df.shape[0] > 0.5
                ):  # More than 50% convertible to numeric
                    issues.append(
                        {
                            "type": "inconsistent_type",
                            "description": f"Column '{col}' contains mostly numeric values but is stored as text",
                            "affected_column": col,
                            "suggested_type": "numeric",
                            "severity": "medium",
                        }
                    )

        return issues

    def _suggest_preprocessing(self, df, df_info, ai_analysis):
        """
        Suggest preprocessing steps based on data quality issues.

        Args:
            df (pandas.DataFrame): The dataset
            df_info (dict): Data profile information
            ai_analysis (dict): AI-generated analysis

        Returns:
            list: Suggested preprocessing steps
        """
        suggestions = []

        # Handle missing values
        missing_cols = [
            col for col, count in df_info["missing_values"].items() if count > 0
        ]
        if missing_cols:
            for col in missing_cols:
                # Suggest appropriate imputation based on column type
                if df[col].dtype in ["int64", "float64"]:
                    suggestions.append(
                        {
                            "type": "missing_values",
                            "method": "imputation",
                            "columns": [col],
                            "parameters": {"strategy": "mean"},
                            "reasoning": f"Column '{col}' has {df_info['missing_values'][col]} missing values. For numeric columns, mean imputation is a standard approach.",
                        }
                    )
                else:
                    suggestions.append(
                        {
                            "type": "missing_values",
                            "method": "imputation",
                            "columns": [col],
                            "parameters": {"strategy": "mode"},
                            "reasoning": f"Column '{col}' has {df_info['missing_values'][col]} missing values. For categorical data, mode imputation (most frequent value) is appropriate.",
                        }
                    )

        # Handle date columns
        for col in df_info.get("date_candidates", []):
            suggestions.append(
                {
                    "type": "date_conversion",
                    "columns": [col],
                    "reasoning": f"Column '{col}' contains date-like values. Converting to datetime enables time-based analysis.",
                }
            )

        # Handle categorical encoding
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if (
                df_info["unique_values"][col] < 10
            ):  # Reasonable number of categories for one-hot
                suggestions.append(
                    {
                        "type": "categorical_encoding",
                        "method": "one_hot",
                        "columns": [col],
                        "reasoning": f"Column '{col}' is categorical with {df_info['unique_values'][col]} unique values. One-hot encoding will make it suitable for analysis.",
                    }
                )
            else:
                suggestions.append(
                    {
                        "type": "categorical_encoding",
                        "method": "label",
                        "columns": [col],
                        "reasoning": f"Column '{col}' is categorical with many unique values ({df_info['unique_values'][col]}). Label encoding will convert it to numeric.",
                    }
                )

        # Add AI-suggested preprocessing if available
        if isinstance(ai_analysis, dict) and "preprocessing_suggestions" in ai_analysis:
            ai_suggestions = ai_analysis["preprocessing_suggestions"]
            if isinstance(ai_suggestions, list):
                for suggestion in ai_suggestions:
                    if isinstance(suggestion, dict) and "type" in suggestion:
                        # Add the AI suggestion if it's not already covered
                        if not any(
                            s["type"] == suggestion["type"]
                            and s.get("columns") == suggestion.get("columns")
                            for s in suggestions
                        ):
                            suggestions.append(suggestion)

        return suggestions
