import streamlit as st
import pandas as pd
import json
import base64
import datetime
import textwrap
import zipfile
import io
import numpy as np
from modules.visualization import create_visualization


def generate_config_json(dashboard_config, visualizations, business_context, analyses):
    """
    Generate a JSON configuration file.

    Args:
        dashboard_config (dict): Dashboard configuration
        visualizations (list): Visualization configurations
        business_context (dict): Business context information
        analyses (list): Analysis results

    Returns:
        str: JSON string
    """

    # Custom JSON encoder to handle pandas Timestamp objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif hasattr(obj, "to_json"):
                return obj.to_json()
            elif pd.isna(obj):
                return None
            else:
                return super(DateTimeEncoder, self).default(obj)

    config = {
        "metadata": {
            "project_name": dashboard_config.get("title", "Dashboard"),
            "created_at": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
        },
        "business_context": business_context,
        "analyses": analyses,
        "visualizations": visualizations,
        "dashboard": dashboard_config,
    }

    return json.dumps(config, indent=2, cls=DateTimeEncoder)


def generate_config_json(dashboard_config, visualizations, business_context, analyses):
    """
    Generate a JSON configuration file.

    Args:
        dashboard_config (dict): Dashboard configuration
        visualizations (list): Visualization configurations
        business_context (dict): Business context information
        analyses (list): Analysis results

    Returns:
        str: JSON string
    """

    # Custom JSON encoder to handle pandas Timestamp objects
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif hasattr(obj, "to_json"):
                return obj.to_json()
            elif pd.isna(obj):
                return None
            else:
                return super(DateTimeEncoder, self).default(obj)

    config = {
        "metadata": {
            "project_name": dashboard_config.get("title", "Dashboard"),
            "created_at": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
        },
        "business_context": business_context,
        "analyses": analyses,
        "visualizations": visualizations,
        "dashboard": dashboard_config,
    }

    return json.dumps(config, indent=2, cls=DateTimeEncoder)


def generate_preprocessing_code(preprocessing_steps):
    """
    Generate code for data preprocessing steps.

    Args:
        preprocessing_steps (list): List of preprocessing steps

    Returns:
        str: Python code for preprocessing
    """
    if not preprocessing_steps:
        return "    # No preprocessing needed\n    return df"

    code = "    # Apply preprocessing steps\n"

    for step in preprocessing_steps:
        step_type = step.get("type")

        if step_type == "missing_values":
            method = step.get("method")
            columns = step.get("columns", [])
            columns_str = str(columns).replace("'", '"')

            if method == "drop_rows":
                code += f"    # Drop rows with missing values in {columns_str}\n"
                code += f"    df = df.dropna(subset={columns_str})\n"

            elif method == "imputation":
                strategy = step.get("parameters", {}).get("strategy", "mean")
                code += (
                    f"    # Impute missing values in {columns_str} using {strategy}\n"
                )

                for col in columns:
                    if strategy == "mean":
                        code += f"    if pd.api.types.is_numeric_dtype(df['{col}']):\n"
                        code += f"        df['{col}'] = df['{col}'].fillna(df['{col}'].mean())\n"
                    elif strategy == "median":
                        code += f"    if pd.api.types.is_numeric_dtype(df['{col}']):\n"
                        code += f"        df['{col}'] = df['{col}'].fillna(df['{col}'].median())\n"
                    elif strategy == "mode":
                        code += f"    if not df['{col}'].empty and not pd.isna(df['{col}']).all():\n"
                        code += f"        df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])\n"
                    elif strategy == "constant":
                        fill_value = step.get("parameters", {}).get("value")
                        code += f"    df['{col}'] = df['{col}'].fillna({repr(fill_value)})\n"

        elif step_type == "date_conversion":
            columns = step.get("columns", [])
            columns_str = str(columns).replace("'", '"')

            code += f"    # Convert columns to datetime: {columns_str}\n"
            for col in columns:
                code += f"    try:\n"
                code += f"        df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')\n"
                code += f"    except Exception as e:\n"
                code += f'        st.warning(f"Error converting {col} to datetime: {{str(e)}}")\n'

        elif step_type == "standardization":
            columns = step.get("columns", [])
            columns_str = str(columns).replace("'", '"')

            code += f"    # Standardize columns: {columns_str}\n"
            for col in columns:
                code += f"    if pd.api.types.is_numeric_dtype(df['{col}']):\n"
                code += f"        mean = df['{col}'].mean()\n"
                code += f"        std = df['{col}'].std()\n"
                code += f"        if std > 0:  # Avoid division by zero\n"
                code += f"            df['{col}'] = (df['{col}'] - mean) / std\n"

        elif step_type == "normalization":
            columns = step.get("columns", [])
            columns_str = str(columns).replace("'", '"')

            code += f"    # Normalize columns: {columns_str}\n"
            for col in columns:
                code += f"    if pd.api.types.is_numeric_dtype(df['{col}']):\n"
                code += f"        min_val = df['{col}'].min()\n"
                code += f"        max_val = df['{col}'].max()\n"
                code += f"        if max_val > min_val:  # Avoid division by zero\n"
                code += f"            df['{col}'] = (df['{col}'] - min_val) / (max_val - min_val)\n"

        elif step_type == "categorical_encoding":
            method = step.get("method")
            columns = step.get("columns", [])
            columns_str = str(columns).replace("'", '"')

            if method == "one_hot":
                code += f"    # One-hot encode columns: {columns_str}\n"
                for col in columns:
                    code += f"    # One-hot encode {col}\n"
                    code += f"    if '{col}' in df.columns:\n"
                    code += f"        dummies = pd.get_dummies(df['{col}'], prefix='{col}')\n"
                    code += f"        df = pd.concat([df, dummies], axis=1)\n"
                    code += f"        df = df.drop('{col}', axis=1)\n"

            elif method == "label":
                code += f"    # Label encode columns: {columns_str}\n"
                for col in columns:
                    code += f"    if '{col}' in df.columns:\n"
                    code += f"        df['{col}_encoded'] = df['{col}'].astype('category').cat.codes\n"

    code += "    return df"
    return code


def generate_streamlit_code(
    dashboard_config, visualizations, business_context, preprocessing_steps, filename
):
    """
    Generate Streamlit code for the dashboard.

    Args:
        dashboard_config (dict): Dashboard configuration
        visualizations (list): Visualization configurations
        business_context (dict): Business context information
        preprocessing_steps (list): Data preprocessing steps
        filename (str): Original data filename

    Returns:
        str: Generated Streamlit code
    """
    # Header and imports
    code = """
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime

# Page configuration
st.set_page_config(
    page_title="{title}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
@st.cache_data
def load_data(filepath="{filename}"):
    \"\"\"
    Load and preprocess data
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pandas.DataFrame: Processed dataframe
    \"\"\"
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            st.error(f"Data file not found: {{filepath}}")
            return pd.DataFrame()
            
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Apply preprocessing steps
{preprocessing_code}
    except Exception as e:
        st.error(f"Error loading data: {{str(e)}}")
        return pd.DataFrame()

# Dashboard title and info
def show_dashboard_info():
    st.title("{title}")
    st.write("{description}")
    
    # Show timestamp
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last updated:** {{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")

# Main function
def main():
    # Show dashboard info
    show_dashboard_info()
    
    # Load data
    df = load_data()
    
    # Check if data is loaded successfully
    if df.empty:
        st.error("No data to display. Please check the data file.")
        return
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Initialize filtered dataframe
    filtered_df = df.copy()
    
    # Add filters
{filters}
    
    # Dashboard layout
{layout}

if __name__ == "__main__":
    main()
""".strip()

    # Format title and description
    code = code.replace("{title}", dashboard_config.get("title", "Dashboard"))
    code = code.replace("{description}", dashboard_config.get("description", ""))
    code = code.replace("{filename}", filename)

    # Insert preprocessing code
    preprocessing_code = generate_preprocessing_code(preprocessing_steps)
    code = code.replace("{preprocessing_code}", preprocessing_code)

    # Generate filters code
    filters_code = ""
    for filter_config in dashboard_config.get("filters", []):
        col = filter_config["column"]
        filter_title = filter_config["title"]
        filter_type = filter_config["type"]

        if filter_type == "select":
            filters_code += f"""
    # {filter_title}
    if "{col}" in df.columns:
        {col}_filter = st.sidebar.selectbox(
            "{filter_title}",
            ["All"] + sorted(df["{col}"].dropna().unique().tolist())
        )
        if {col}_filter != "All":
            filtered_df = filtered_df[filtered_df["{col}"] == {col}_filter]
"""
        elif filter_type == "multiselect":
            filters_code += f"""
    # {filter_title}
    if "{col}" in df.columns:
        {col}_filter = st.sidebar.multiselect(
            "{filter_title}",
            sorted(df["{col}"].dropna().unique().tolist()),
            default=[]
        )
        if {col}_filter:
            filtered_df = filtered_df[filtered_df["{col}"].isin({col}_filter)]
"""
        elif filter_type == "slider":
            filters_code += f"""
    # {filter_title}
    if "{col}" in df.columns and pd.api.types.is_numeric_dtype(df["{col}"]):
        min_{col}, max_{col} = st.sidebar.slider(
            "{filter_title}",
            float(df["{col}"].min()),
            float(df["{col}"].max()),
            (float(df["{col}"].min()), float(df["{col}"].max()))
        )
        filtered_df = filtered_df[(filtered_df["{col}"] >= min_{col}) & (filtered_df["{col}"] <= max_{col})]
"""

    code = code.replace("{filters}", filters_code.strip())

    # Generate layout code
    layout_code = ""

    for i, section_vizs in enumerate(dashboard_config.get("layout", [])):
        if section_vizs:
            # Create row
            layout_code += f"""
    # Row {i + 1}
    row{i + 1}_cols = st.columns({len(section_vizs)})
"""

            # Add visualizations to row
            for j, viz_id in enumerate(section_vizs):
                # Find visualization config
                viz_config = next(
                    (v for v in visualizations if v["id"] == viz_id), None
                )

                if viz_config:
                    viz_type = viz_config.get("type")
                    viz_title = viz_config.get("title", f"Visualization {j + 1}")
                    data_config = viz_config.get("data_config", {})

                    layout_code += f"""
    with row{i + 1}_cols[{j}]:
        st.subheader("{viz_title}")
"""

                    # Generate code for different visualization types
                    if viz_type == "line_chart":
                        x = data_config.get("x")
                        y = data_config.get("y", [])

                        if not isinstance(y, list):
                            y = [y]

                        y_str = str(y).replace("'", '"')

                        layout_code += f"""
        # Check if all required columns exist
        if "{x}" in filtered_df.columns and all(col in filtered_df.columns for col in {y_str}):
            fig = px.line(
                filtered_df, 
                x="{x}", 
                y={y_str}, 
                title="{viz_title}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Some columns required for this visualization are missing.")
"""

                    elif viz_type == "bar_chart":
                        x = data_config.get("x")
                        y = data_config.get("y", [])

                        if not isinstance(y, list):
                            y = [y]

                        y_str = str(y).replace("'", '"')

                        layout_code += f"""
        # Check if all required columns exist
        if "{x}" in filtered_df.columns and all(col in filtered_df.columns for col in {y_str}):
            fig = px.bar(
                filtered_df, 
                x="{x}", 
                y={y_str}, 
                title="{viz_title}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Some columns required for this visualization are missing.")
"""

                    elif viz_type == "scatter_plot":
                        x = data_config.get("x")
                        y = data_config.get("y")

                        layout_code += f"""
        # Check if all required columns exist
        if "{x}" in filtered_df.columns and "{y}" in filtered_df.columns:
            fig = px.scatter(
                filtered_df, 
                x="{x}", 
                y="{y}", 
                title="{viz_title}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Some columns required for this visualization are missing.")
"""

                    elif viz_type == "histogram":
                        x = data_config.get("x")

                        layout_code += f"""
        # Check if column exists
        if "{x}" in filtered_df.columns:
            fig = px.histogram(
                filtered_df, 
                x="{x}", 
                title="{viz_title}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Column '{x}' is missing.")
"""

                    elif viz_type == "pie_chart":
                        values = data_config.get("values")
                        names = data_config.get("names")

                        layout_code += f"""
        # Check if all required columns exist
        if "{values}" in filtered_df.columns and "{names}" in filtered_df.columns:
            fig = px.pie(
                filtered_df, 
                values="{values}", 
                names="{names}", 
                title="{viz_title}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Some columns required for this visualization are missing.")
"""

                    elif viz_type == "heatmap":
                        # For heatmap, we'll need to generate a dynamic correlation matrix
                        layout_code += f"""
        # Generate correlation matrix from numeric columns
        numeric_df = filtered_df.select_dtypes(include=['number'])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().round(2)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmid=0
            ))
            fig.update_layout(title="{viz_title}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns to generate correlation matrix.")
"""

                    elif viz_type == "kpi_card":
                        metric = data_config.get("metric", "")

                        layout_code += f"""
        # Check if column exists
        if "{metric}" in filtered_df.columns:
            try:
                # Calculate key metrics
                metric_value = filtered_df["{metric}"].mean()
                metric_min = filtered_df["{metric}"].min()
                metric_max = filtered_df["{metric}"].max()
                
                # Display in KPI card format
                st.metric(
                    "{metric}",
                    f"{{metric_value:.2f}}",
                    f"Range: {{metric_min:.2f}} to {{metric_max:.2f}}"
                )
                
            except Exception as e:
                st.warning(f"Error calculating KPI: {{str(e)}}")
        else:
            st.warning("Column '{metric}' is missing.")
"""

    code = code.replace("{layout}", layout_code.strip())

    return code


def generate_zip_file(
    code, config_json, dashboard_config, preprocessing_steps, df, filename
):
    """
    Generate a ZIP file containing the dashboard code and configuration.

    Args:
        code (str): Generated Streamlit code
        config_json (str): Dashboard configuration JSON
        dashboard_config (dict): Dashboard configuration
        preprocessing_steps (list): Preprocessing steps
        df (pandas.DataFrame): Original dataframe
        filename (str): Original filename

    Returns:
        bytes: ZIP file as bytes
    """
    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add app.py
        zipf.writestr("app.py", code)

        # Add README.md
        readme = f"""
# {dashboard_config.get("title", "Dashboard")}

{dashboard_config.get("description", "")}

## Getting Started

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```
   streamlit run app.py
   ```

## Data Updates

To update the dashboard with new data:

1. Replace the `{filename}` file with your updated CSV file
2. Make sure the new CSV has the same column structure as the original
3. Run the dashboard as usual

## Configuration

The dashboard configuration is stored in `dashboard_config.json`.
        """.strip()

        zipf.writestr("README.md", readme)

        # Add configuration JSON
        zipf.writestr("dashboard_config.json", config_json)

        # Add requirements.txt
        requirements = """
streamlit==1.31.1
pandas==2.2.0
numpy==1.26.3
plotly==5.18.0
        """.strip()

        zipf.writestr("requirements.txt", requirements)

        # Add the original data as CSV
        try:
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zipf.writestr(filename, csv_buffer.getvalue())
        except Exception as e:
            print(f"Error adding CSV to ZIP: {str(e)}")

        # Add preprocessing documentation
        if preprocessing_steps:
            preprocessing_doc = "# Data Preprocessing Steps\n\n"

            for i, step in enumerate(preprocessing_steps):
                step_type = step.get("type")
                preprocessing_doc += (
                    f"## Step {i + 1}: {step_type.replace('_', ' ').title()}\n\n"
                )

                if step_type == "missing_values":
                    method = step.get("method")
                    columns = step.get("columns", [])
                    preprocessing_doc += f"- Method: {method}\n"
                    preprocessing_doc += f"- Applied to columns: {', '.join(columns)}\n"

                    if method == "imputation":
                        strategy = step.get("parameters", {}).get("strategy", "mean")
                        preprocessing_doc += f"- Strategy: {strategy}\n"

                elif step_type == "date_conversion":
                    columns = step.get("columns", [])
                    preprocessing_doc += (
                        f"- Converted to datetime: {', '.join(columns)}\n"
                    )

                elif step_type == "standardization":
                    columns = step.get("columns", [])
                    preprocessing_doc += (
                        f"- Standardized columns: {', '.join(columns)}\n"
                    )

                elif step_type == "normalization":
                    columns = step.get("columns", [])
                    preprocessing_doc += f"- Normalized columns: {', '.join(columns)}\n"

                elif step_type == "categorical_encoding":
                    method = step.get("method")
                    columns = step.get("columns", [])
                    preprocessing_doc += f"- Encoding method: {method}\n"
                    preprocessing_doc += f"- Applied to columns: {', '.join(columns)}\n"

                preprocessing_doc += "\n"

            zipf.writestr("preprocessing_documentation.md", preprocessing_doc)

    # Reset pointer to start of buffer
    zip_buffer.seek(0)

    return zip_buffer.getvalue()


def render_export_interface(
    dashboard_config, visualizations, business_context, analyses
):
    """
    Render the export interface.

    Args:
        dashboard_config (dict): Dashboard configuration
        visualizations (list): Visualization configurations
        business_context (dict): Business context information
        analyses (list): Analysis results

    Returns:
        bool: Whether the export was completed
    """
    st.header("Export Dashboard")

    if not dashboard_config:
        st.error(
            "No dashboard configuration available. Please complete the Dashboard Assembly step first."
        )
        if st.button("Go to Dashboard Assembly"):
            st.session_state.step = "dashboard_assembly"
            st.rerun()
        return False

    st.write(
        "Your dashboard is ready to export. You can download the code and configuration files."
    )

    # Get preprocessing steps
    preprocessing_steps = st.session_state.preprocessing

    # Get original filename
    filename = st.session_state.dataset_name or "data.csv"

    # Generate Streamlit code
    code = generate_streamlit_code(
        dashboard_config,
        visualizations,
        business_context,
        preprocessing_steps,
        filename,
    )

    # Generate configuration JSON
    config_json = generate_config_json(
        dashboard_config, visualizations, business_context, analyses
    )

    # Display tabs for code, configuration, and preview
    tab1, tab2, tab3 = st.tabs(["Dashboard Code", "Configuration", "Download"])

    with tab1:
        st.subheader("Generated Streamlit Code")
        st.code(code, language="python")

        # Option to download just the code
        if st.button("Download Code"):
            st.download_button(
                label="Download app.py",
                data=code,
                file_name="app.py",
                mime="text/plain",
            )

    with tab2:
        st.subheader("Dashboard Configuration")
        st.json(json.loads(config_json))

        # Option to download just the configuration
        if st.button("Download Configuration"):
            st.download_button(
                label="Download dashboard_config.json",
                data=config_json,
                file_name="dashboard_config.json",
                mime="application/json",
            )

    with tab3:
        st.subheader("Download Complete Package")
        st.write(
            "Download a ZIP file containing everything you need to run and update the dashboard:"
        )

        # Generate ZIP file
        zip_data = generate_zip_file(
            code,
            config_json,
            dashboard_config,
            preprocessing_steps,
            st.session_state.dataset,
            filename,
        )

        # Create download button
        st.download_button(
            label="Download Dashboard Package",
            data=zip_data,
            file_name=f"{dashboard_config.get('title', 'dashboard').lower().replace(' ', '_')}_package.zip",
            mime="application/zip",
            help="Download a ZIP file containing the Streamlit app, configuration, and data",
        )

        st.markdown("""
        ### What's included in the package:
        
        - **app.py**: The main Streamlit application code with preprocessing logic
        - **dashboard_config.json**: Configuration file with dashboard settings
        - **README.md**: Instructions for running and updating the dashboard
        - **requirements.txt**: Required Python packages
        - **Original data file**: Your uploaded data in CSV format
        - **preprocessing_documentation.md**: Details of applied preprocessing steps
        """)

        st.markdown("""
        ### Updating the Dashboard:
        
        To update the dashboard with new data:
        
        1. Replace the CSV file with your updated data file
        2. Make sure the columns match the original structure
        3. Run the dashboard with `streamlit run app.py`
        
        The preprocessing steps will be automatically applied to the new data.
        """)

    # Complete button
    if st.button("Complete Export"):
        st.success("Dashboard export completed successfully!")
        return True

    return False
