import streamlit as st
import pandas as pd
import json
import base64
import datetime
import textwrap
import zipfile
import io
from modules.visualization import create_visualization


def generate_streamlit_code(dashboard_config, visualizations, business_context):
    """
    Generate Streamlit code for the dashboard.

    Args:
        dashboard_config (dict): Dashboard configuration
        visualizations (list): Visualization configurations
        business_context (dict): Business context information

    Returns:
        str: Generated Streamlit code
    """
    # Header and imports
    code = """
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="{title}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
@st.cache_data
def load_data():
    # Replace this with your data loading logic
    # For example: return pd.read_csv("your_data.csv")
    data = {data_placeholder}
    return pd.DataFrame(data)

# Main function
def main():
    # Dashboard title
    st.title("{title}")
    st.write("{description}")
    
    # Load data
    df = load_data()
    
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

    # Generate filters code
    filters_code = ""
    for filter_config in dashboard_config.get("filters", []):
        col = filter_config["column"]
        filter_title = filter_config["title"]
        filter_type = filter_config["type"]

        if filter_type == "select":
            filters_code += textwrap.indent(
                f"""
    # {filter_title}
    {col}_filter = st.sidebar.selectbox(
        "{filter_title}",
        ["All"] + sorted(df["{col}"].dropna().unique().tolist())
    )
    if {col}_filter != "All":
        filtered_df = filtered_df[filtered_df["{col}"] == {col}_filter]
            """,
                "    ",
            )
        elif filter_type == "multiselect":
            filters_code += textwrap.indent(
                f"""
    # {filter_title}
    {col}_filter = st.sidebar.multiselect(
        "{filter_title}",
        sorted(df["{col}"].dropna().unique().tolist()),
        default=[]
    )
    if {col}_filter:
        filtered_df = filtered_df[filtered_df["{col}"].isin({col}_filter)]
            """,
                "    ",
            )
        elif filter_type == "slider":
            filters_code += textwrap.indent(
                f"""
    # {filter_title}
    min_{col}, max_{col} = st.sidebar.slider(
        "{filter_title}",
        float(df["{col}"].min()),
        float(df["{col}"].max()),
        (float(df["{col}"].min()), float(df["{col}"].max()))
    )
    filtered_df = filtered_df[(filtered_df["{col}"] >= min_{col}) & (filtered_df["{col}"] <= max_{col})]
            """,
                "    ",
            )

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
        fig = px.line(
            filtered_df, 
            x="{x}", 
            y={y_str}, 
            title="{viz_title}"
        )
        st.plotly_chart(fig, use_container_width=True)
                        """

                    elif viz_type == "bar_chart":
                        x = data_config.get("x")
                        y = data_config.get("y", [])

                        if not isinstance(y, list):
                            y = [y]

                        y_str = str(y).replace("'", '"')

                        layout_code += f"""
        fig = px.bar(
            filtered_df, 
            x="{x}", 
            y={y_str}, 
            title="{viz_title}"
        )
        st.plotly_chart(fig, use_container_width=True)
                        """

                    elif viz_type == "scatter_plot":
                        x = data_config.get("x")
                        y = data_config.get("y")

                        layout_code += f"""
        fig = px.scatter(
            filtered_df, 
            x="{x}", 
            y="{y}", 
            title="{viz_title}"
        )
        st.plotly_chart(fig, use_container_width=True)
                        """

                    elif viz_type == "histogram":
                        x = data_config.get("x")

                        layout_code += f"""
        fig = px.histogram(
            filtered_df, 
            x="{x}", 
            title="{viz_title}"
        )
        st.plotly_chart(fig, use_container_width=True)
                        """

                    elif viz_type == "pie_chart":
                        values = data_config.get("values")
                        names = data_config.get("names")

                        layout_code += f"""
        fig = px.pie(
            filtered_df, 
            values="{values}", 
            names="{names}", 
            title="{viz_title}"
        )
        st.plotly_chart(fig, use_container_width=True)
                        """

                    elif viz_type == "heatmap":
                        # For heatmap, we'll need to generate a dynamic correlation matrix
                        layout_code += f"""
        # Generate correlation matrix
        numeric_df = filtered_df.select_dtypes(include=['number'])
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
                        """

    code = code.replace("{layout}", layout_code.strip())

    # Generate data placeholder with sample records
    try:
        sample_data = {
            "column1": [1, 2, 3],
            "column2": ["A", "B", "C"],
            "column3": [10.1, 20.2, 30.3],
        }

        # Replace with more meaningful placeholder based on business context
        if (
            business_context
            and "dimensions" in business_context
            and "key_metrics" in business_context
        ):
            sample_data = {}

            # Add dimensions
            for dim in business_context.get("dimensions", [])[:2]:
                sample_data[dim] = ["Value1", "Value2", "Value3"]

            # Add metrics
            for metric in business_context.get("key_metrics", [])[:3]:
                sample_data[metric] = [100, 200, 300]

        data_placeholder = json.dumps(sample_data, indent=4)
    except:
        data_placeholder = "{}"

    code = code.replace("{data_placeholder}", data_placeholder)

    return code


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


def generate_zip_file(code, config_json, dashboard_config):
    """
    Generate a ZIP file containing the dashboard code and configuration.

    Args:
        code (str): Generated Streamlit code
        config_json (str): Dashboard configuration JSON
        dashboard_config (dict): Dashboard configuration

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
   pip install streamlit pandas plotly
   ```

2. Run the dashboard:
   ```
   streamlit run app.py
   ```

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
plotly==5.18.0
        """.strip()

        zipf.writestr("requirements.txt", requirements)

    # Reset pointer to start of buffer
    zip_buffer.seek(0)

    return zip_buffer.getvalue()


def create_download_link(file_content, file_name, file_type):
    """
    Create a download link for a file.

    Args:
        file_content: File content as bytes
        file_name (str): File name
        file_type (str): File MIME type

    Returns:
        str: HTML download link
    """
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:{file_type};base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href


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

    # Generate Streamlit code
    code = generate_streamlit_code(dashboard_config, visualizations, business_context)

    # Generate configuration JSON
    config_json = generate_config_json(
        dashboard_config, visualizations, business_context, analyses
    )

    # Display tabs for code and configuration
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
            "Download a ZIP file containing everything you need to run the dashboard:"
        )

        # Generate ZIP file
        zip_data = generate_zip_file(code, config_json, dashboard_config)

        # Create download button
        st.download_button(
            label="Download Dashboard Package",
            data=zip_data,
            file_name=f"{dashboard_config.get('title', 'dashboard').lower().replace(' ', '_')}_package.zip",
            mime="application/zip",
            help="Download a ZIP file containing the Streamlit app, configuration, and instructions",
        )

        st.markdown("""
        ### What's included in the package:
        
        - **app.py**: The main Streamlit application code
        - **dashboard_config.json**: Configuration file with dashboard settings
        - **README.md**: Instructions for running the dashboard
        - **requirements.txt**: Required Python packages
        """)

        st.markdown("""
        ### Next Steps:
        
        1. Extract the ZIP file to a directory
        2. Install the required packages with `pip install -r requirements.txt`
        3. Run the dashboard with `streamlit run app.py`
        4. Customize the code as needed for your specific data source
        """)

    # Complete button
    if st.button("Complete Export"):
        st.success("Dashboard export completed successfully!")
        return True

    return False
