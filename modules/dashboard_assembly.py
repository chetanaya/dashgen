import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from modules.visualization import create_visualization


def render_dashboard_assembly(df, visualizations, business_context):
    """
    Render the dashboard assembly interface.

    Args:
        df (pandas.DataFrame): Dataset
        visualizations (list): Selected visualizations
        business_context (dict): Business context information

    Returns:
        dict: Dashboard configuration
    """
    st.header("Dashboard Assembly")

    if not visualizations:
        st.error(
            "No visualizations available. Please complete the visualization step first."
        )
        if st.button("Go to Visualization Generation"):
            st.session_state.step = "visualization_generation"
            st.rerun()
        return None

    # Initialize dashboard config if not already in session
    if (
        "dashboard_config" not in st.session_state
        or not st.session_state.dashboard_config
    ):
        st.session_state.dashboard_config = {
            "title": f"{business_context.get('domain', 'Data')} Dashboard",
            "description": f"Dashboard for analyzing {business_context.get('domain', 'data')}",
            "layout": [],
            "filters": [],
        }

    # Tab for configuration and preview
    tab1, tab2 = st.tabs(["Dashboard Configuration", "Preview"])

    with tab1:
        # Dashboard general settings
        st.subheader("Dashboard Settings")

        # Title and description
        title = st.text_input(
            "Dashboard Title", value=st.session_state.dashboard_config.get("title", "")
        )

        description = st.text_area(
            "Dashboard Description",
            value=st.session_state.dashboard_config.get("description", ""),
            height=100,
        )

        # Update config
        if title != st.session_state.dashboard_config.get("title", ""):
            st.session_state.dashboard_config["title"] = title

        if description != st.session_state.dashboard_config.get("description", ""):
            st.session_state.dashboard_config["description"] = description

        # Layout organization
        st.subheader("Layout Organization")

        st.write("Drag and drop visualizations to organize your dashboard layout.")

        # For POC, we'll use a simpler approach without actual drag-and-drop
        st.write("Assign visualizations to sections of your dashboard:")

        # Define layout sections
        layout_sections = ["Top Row", "Middle Row", "Bottom Row"]

        # Initialize layout in config if not present
        if (
            "layout" not in st.session_state.dashboard_config
            or not st.session_state.dashboard_config["layout"]
        ):
            st.session_state.dashboard_config["layout"] = [[] for _ in layout_sections]

        # Ensure layout has enough sections
        while len(st.session_state.dashboard_config["layout"]) < len(layout_sections):
            st.session_state.dashboard_config["layout"].append([])

        # Create a selection interface for each section
        for i, section in enumerate(layout_sections):
            st.write(f"**{section}**")

            # Get current visualizations in this section
            current_viz_ids = st.session_state.dashboard_config["layout"][i]

            # Display available visualizations for selection
            options = ["None"] + [v["title"] for v in visualizations]
            selected_indices = []

            # Find indices of currently selected visualizations
            for viz_id in current_viz_ids:
                for j, viz in enumerate(visualizations):
                    if viz["id"] == viz_id:
                        selected_indices.append(j + 1)  # +1 for "None" option
                        break

            # Multi-select for visualizations
            selected = st.multiselect(
                f"Select visualizations for {section}",
                options,
                default=[options[idx] for idx in selected_indices]
                if selected_indices
                else ["None"],
                key=f"select_{section}",
            )

            # Update layout config
            new_viz_ids = []
            for title in selected:
                if title != "None":
                    # Find visualization ID by title
                    for viz in visualizations:
                        if viz["title"] == title:
                            new_viz_ids.append(viz["id"])
                            break

            st.session_state.dashboard_config["layout"][i] = new_viz_ids

        # Filter configuration
        st.subheader("Filter Configuration")

        # Get dimensions from business context
        dimensions = business_context.get("dimensions", [])

        if dimensions:
            st.write("Add filters to make your dashboard interactive:")

            # Initialize filters in config if not present
            if "filters" not in st.session_state.dashboard_config:
                st.session_state.dashboard_config["filters"] = []

            # Create selection interface for filters
            selected_dims = st.multiselect(
                "Select dimensions to use as filters",
                dimensions,
                default=[
                    f["column"]
                    for f in st.session_state.dashboard_config.get("filters", [])
                ],
            )

            # Update filters in config
            new_filters = []
            for dim in selected_dims:
                # Check if filter already exists
                existing = next(
                    (
                        f
                        for f in st.session_state.dashboard_config.get("filters", [])
                        if f["column"] == dim
                    ),
                    None,
                )

                if existing:
                    new_filters.append(existing)
                else:
                    # Create new filter config
                    filter_type = "select" if df[dim].nunique() < 10 else "multiselect"
                    new_filters.append(
                        {
                            "column": dim,
                            "type": filter_type,
                            "title": f"Filter by {dim}",
                            "default": "All",
                        }
                    )

            st.session_state.dashboard_config["filters"] = new_filters
        else:
            st.info("No dimensions available for filtering.")

    with tab2:
        # Preview the dashboard
        st.subheader("Dashboard Preview")

        # Display dashboard title and description
        st.title(st.session_state.dashboard_config.get("title", "Dashboard"))
        st.write(st.session_state.dashboard_config.get("description", ""))

        # Display filters
        if st.session_state.dashboard_config.get("filters"):
            st.subheader("Filters")

            # Create filter widgets (for demonstration, they won't actually filter the data)
            cols = st.columns(min(len(st.session_state.dashboard_config["filters"]), 3))

            for i, filter_config in enumerate(
                st.session_state.dashboard_config["filters"]
            ):
                col_idx = i % len(cols)

                with cols[col_idx]:
                    col = filter_config["column"]

                    if filter_config["type"] == "select":
                        st.selectbox(
                            filter_config["title"],
                            ["All"] + df[col].dropna().unique().tolist(),
                            key=f"preview_filter_{col}",
                        )
                    elif filter_config["type"] == "multiselect":
                        st.multiselect(
                            filter_config["title"],
                            df[col].dropna().unique().tolist(),
                            default=[],
                            key=f"preview_filter_{col}",
                        )
                    elif filter_config[
                        "type"
                    ] == "slider" and pd.api.types.is_numeric_dtype(df[col]):
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        st.slider(
                            filter_config["title"],
                            min_val,
                            max_val,
                            (min_val, max_val),
                            key=f"preview_filter_{col}",
                        )

        # Display visualizations according to layout
        for section_vizs in st.session_state.dashboard_config["layout"]:
            if section_vizs:
                # Create columns based on number of visualizations in this section
                cols = st.columns(len(section_vizs))

                for i, viz_id in enumerate(section_vizs):
                    # Find visualization config
                    viz_config = next(
                        (v for v in visualizations if v["id"] == viz_id), None
                    )

                    if viz_config:
                        with cols[i]:
                            st.subheader(viz_config["title"])

                            # Create visualization
                            fig = create_visualization(df, viz_config)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(
                                    f"Could not create visualization: {viz_config['title']}"
                                )

    # Save button
    if st.button("Save Dashboard and Continue to Export"):
        # Return the dashboard configuration
        return st.session_state.dashboard_config

    return None
