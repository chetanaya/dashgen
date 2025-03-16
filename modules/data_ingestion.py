import pandas as pd
import streamlit as st
from utils.data_utils import load_csv
from agents.data_profiler import DataProfilerAgent
from config.settings import MAX_FILE_SIZE_MB


def render_upload_interface():
    """
    Render the CSV upload interface.

    Returns:
        pandas.DataFrame or None: Loaded DataFrame if upload successful, None otherwise
    """
    st.header("Data Upload")
    st.markdown(
        """
    Upload your CSV file to begin creating your intelligent dashboard.
    
    **Supported file type:** CSV
    **Maximum file size:** {0} MB
    """.format(MAX_FILE_SIZE_MB)
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB. Please upload a smaller file."
            )
            return None

        try:
            with st.spinner("Loading data..."):
                # If file is large, add sampling option
                sample_data = False
                if file_size_mb > 5:  # Show sampling option for files > 5MB
                    sample_data = st.checkbox(
                        "Sample data for faster processing",
                        value=True,
                        help="Enable to process a representative sample of large datasets",
                    )

                # Load the data
                df = load_csv(uploaded_file, sample=sample_data)

                # Show success message
                st.success(
                    f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns."
                )

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                return df, uploaded_file.name

        except Exception as e:
            st.error(f"Error loading the file: {str(e)}")
            return None

    return None


def run_data_profiling(df):
    """
    Run the data profiling agent on the uploaded dataset.

    Args:
        df (pandas.DataFrame): The dataset to profile

    Returns:
        dict: The complete data profile
    """
    with st.spinner("Analyzing your dataset..."):
        profiler = DataProfilerAgent()
        profile = profiler.profile_dataset(df)
        return profile


def display_data_profile(profile):
    """
    Display the data profile information in an organized way.

    Args:
        profile (dict): The data profile to display
    """
    st.header("Data Profile")

    # Basic info in expander
    with st.expander("Dataset Overview", expanded=True):
        st.write(f"**Rows:** {profile['basic_info']['rows']}")
        st.write(f"**Columns:** {profile['basic_info']['columns']}")
        st.write(f"**Memory Usage:** {profile['basic_info']['memory_usage']:.2f} MB")

        # Display found data types
        st.subheader("Column Types")
        type_df = pd.DataFrame(
            {
                "Column": list(profile["column_types"].keys()),
                "Type": list(profile["column_types"].values()),
            }
        )
        st.dataframe(type_df)

    # AI Analysis
    with st.expander("AI Analysis", expanded=True):
        if isinstance(profile.get("ai_analysis"), dict):
            ai_analysis = profile["ai_analysis"]

            # Check for error message
            if "error" in ai_analysis:
                st.warning(f"AI analysis limited: {ai_analysis['error']}")
                st.info(
                    "Set up your OpenAI API key in the .env file for enhanced analysis."
                )

            # Dataset purpose
            if "dataset_purpose" in ai_analysis:
                st.subheader("Dataset Purpose")
                st.write(ai_analysis["dataset_purpose"])

            # Column descriptions
            if "column_descriptions" in ai_analysis:
                st.subheader("Column Descriptions")
                for col, desc in ai_analysis["column_descriptions"].items():
                    st.markdown(f"**{col}**: {desc}")
        else:
            st.warning(
                "AI analysis not available. Please check your OpenAI API key configuration."
            )

    # Data Quality
    with st.expander("Data Quality Issues"):
        if profile.get("quality_issues"):
            for issue in profile["quality_issues"]:
                severity_color = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(
                    issue.get("severity"), "⚪️"
                )

                st.markdown(
                    f"{severity_color} **{issue['type'].replace('_', ' ').title()}**: {issue['description']}"
                )
        else:
            st.write("No significant data quality issues detected.")

    # Preprocessing Suggestions
    with st.expander("Preprocessing Suggestions"):
        if profile.get("suggested_preprocessing"):
            for i, suggestion in enumerate(profile["suggested_preprocessing"]):
                st.markdown(
                    f"**Suggestion {i + 1}**: {suggestion['type'].replace('_', ' ').title()}"
                )
                st.markdown(f"- **Method**: {suggestion.get('method', 'N/A')}")
                st.markdown(
                    f"- **Columns**: {', '.join(suggestion.get('columns', []))}"
                )
                if suggestion.get("reasoning"):
                    st.markdown(f"- **Reasoning**: {suggestion.get('reasoning')}")
                st.markdown("---")
        else:
            st.write("No preprocessing suggestions.")

    # Potential Analyses
    with st.expander("Potential Analyses"):
        if isinstance(profile.get("potential_analyses"), list) and profile.get(
            "potential_analyses"
        ):
            for i, analysis in enumerate(profile["potential_analyses"]):
                st.write(f"{i + 1}. {analysis}")
        else:
            st.write("No potential analyses suggested.")
