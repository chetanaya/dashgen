import streamlit as st
import pandas as pd
from utils.data_utils import apply_preprocessing


def render_preprocessing_interface(profile, df):
    """
    Render the preprocessing interface based on profile suggestions.

    Args:
        profile (dict): Data profile with suggestions
        df (pandas.DataFrame): Original DataFrame

    Returns:
        list: Applied preprocessing steps
    """
    st.header("Data Preprocessing")

    if not profile.get("suggested_preprocessing"):
        st.info("No preprocessing suggestions available for this dataset.")
        if st.button("Continue without preprocessing"):
            return []
        return None

    st.write("Based on the analysis, we recommend the following preprocessing steps:")

    # Display and let user select preprocessing steps
    selected_steps = []

    for i, suggestion in enumerate(profile.get("suggested_preprocessing", [])):
        step_type = suggestion.get("type", "unknown")
        method = suggestion.get("method", "")
        columns = suggestion.get("columns", [])
        reasoning = suggestion.get("reasoning", "")

        # Create a unique key for each suggestion
        key = f"prep_{step_type}_{i}"

        # Create a container for this suggestion
        with st.expander(
            f"{step_type.replace('_', ' ').title()} {method.capitalize() if method else ''}",
            expanded=True,
        ):
            st.markdown(f"**Suggested for columns**: {', '.join(columns)}")
            st.markdown(f"**Reasoning**: {reasoning}")

            # Allow customization based on step type
            if step_type == "missing_values":
                selected = st.checkbox("Apply this suggestion", key=f"{key}_select")
                if selected:
                    # Allow method selection
                    method_options = ["imputation", "drop_rows"]
                    selected_method = st.selectbox(
                        "Method",
                        method_options,
                        index=method_options.index(method)
                        if method in method_options
                        else 0,
                        key=f"{key}_method",
                    )

                    # Additional parameters based on method
                    parameters = {}
                    if selected_method == "imputation":
                        strategy_options = ["mean", "median", "mode", "constant"]
                        strategy = st.selectbox(
                            "Strategy", strategy_options, key=f"{key}_strategy"
                        )
                        parameters["strategy"] = strategy

                        if strategy == "constant":
                            parameters["value"] = st.text_input(
                                "Fill value", key=f"{key}_value"
                            )

                    # Append the selected step
                    selected_steps.append(
                        {
                            "type": step_type,
                            "method": selected_method,
                            "columns": columns,
                            "parameters": parameters,
                        }
                    )

            elif step_type == "date_conversion":
                selected = st.checkbox("Apply this suggestion", key=f"{key}_select")
                if selected:
                    selected_steps.append({"type": step_type, "columns": columns})

            elif step_type == "categorical_encoding":
                selected = st.checkbox("Apply this suggestion", key=f"{key}_select")
                if selected:
                    # Allow method selection
                    method_options = ["one_hot", "label"]
                    selected_method = st.selectbox(
                        "Method",
                        method_options,
                        index=method_options.index(method)
                        if method in method_options
                        else 0,
                        key=f"{key}_method",
                    )

                    # Append the selected step
                    selected_steps.append(
                        {
                            "type": step_type,
                            "method": selected_method,
                            "columns": columns,
                        }
                    )

            elif step_type in ["standardization", "normalization"]:
                selected = st.checkbox("Apply this suggestion", key=f"{key}_select")
                if selected:
                    selected_steps.append({"type": step_type, "columns": columns})

    # Preview button
    if st.button("Preview Changes") and selected_steps:
        # Store selected steps in session state for access after preview
        st.session_state.preview_steps = selected_steps

        with st.spinner("Applying preprocessing..."):
            # Apply preprocessing to get a preview
            preview_df = apply_preprocessing(df, selected_steps)

            # Show the preview
            st.subheader("Preview of Processed Data")
            st.dataframe(preview_df.head(10))

            # Show differences
            st.subheader("Changes Summary")
            for step in selected_steps:
                st.markdown(
                    f"- **{step['type'].replace('_', ' ').title()}** applied to {len(step['columns'])} column(s)"
                )

            # Before/After stats for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                st.subheader("Before/After Statistics")

                # Create comparison for select columns
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    if col in preview_df.columns:
                        st.write(f"**Column: {col}**")

                        # Create before/after comparison
                        before_stats = pd.DataFrame(
                            {
                                "Before": [
                                    df[col].isna().sum(),
                                    df[col].mean()
                                    if pd.api.types.is_numeric_dtype(df[col])
                                    else "N/A",
                                    df[col].std()
                                    if pd.api.types.is_numeric_dtype(df[col])
                                    else "N/A",
                                ]
                            },
                            index=["Missing Values", "Mean", "Std Dev"],
                        )

                        after_stats = pd.DataFrame(
                            {
                                "After": [
                                    preview_df[col].isna().sum(),
                                    preview_df[col].mean()
                                    if pd.api.types.is_numeric_dtype(preview_df[col])
                                    else "N/A",
                                    preview_df[col].std()
                                    if pd.api.types.is_numeric_dtype(preview_df[col])
                                    else "N/A",
                                ]
                            },
                            index=["Missing Values", "Mean", "Std Dev"],
                        )

                        # Display side by side
                        comparison = pd.concat([before_stats, after_stats], axis=1)
                        st.dataframe(comparison)

            # Success message
            st.success("Preprocessing preview generated. Review the changes above.")

    # Confirmation button (outside of preview section)
    if "preview_steps" in st.session_state and st.session_state.preview_steps:
        if st.button("Apply Changes and Continue"):
            return st.session_state.preview_steps

    # Skip button
    if st.button("Skip Preprocessing"):
        return []

    return None
