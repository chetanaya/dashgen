import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid


def generate_visualization_config(analysis_results, business_context):
    """
    Generate visualization configurations based on analysis results.

    Args:
        analysis_results (list): Results from analyses
        business_context (dict): Business context information

    Returns:
        list: Visualization configurations
    """
    visualizations = []

    # Process each analysis result
    for result in analysis_results:
        # Skip if no data
        if result.get("data") is None:
            continue

        analysis_type = result.get("type")

        # Create visualization based on analysis type
        if analysis_type == "descriptive":
            # For descriptive stats, create a KPI card for each metric
            metrics = [col for col in result["data"].index]

            for metric in metrics:
                # Extract stats
                stats = result["data"].loc[metric]

                visualizations.append(
                    {
                        "id": f"kpi_{metric.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}",
                        "type": "kpi_card",
                        "title": f"{metric} Overview",
                        "data_config": {
                            "metric": metric,
                            "value": stats["mean"],
                            "min": stats["min"],
                            "max": stats["max"],
                            "std": stats["std"],
                        },
                        "analysis_id": result.get("title"),
                        "layout_weight": 1,
                    }
                )

        elif analysis_type == "time_series":
            # For time series, create a line chart
            data = result.get("data")
            time_col = data.columns[0]  # First column is time
            metrics = data.columns[1:]  # Rest are metrics

            visualizations.append(
                {
                    "id": f"timeseries_{uuid.uuid4().hex[:6]}",
                    "type": "line_chart",
                    "title": "Trends Over Time",
                    "data_config": {
                        "x": time_col,
                        "y": list(metrics),
                        "data": data.to_dict("records"),
                    },
                    "visual_config": {
                        "color_discrete_sequence": px.colors.qualitative.G10,
                        "markers": True,
                    },
                    "analysis_id": result.get("title"),
                    "layout_weight": 3,
                }
            )

        elif analysis_type == "correlation":
            # For correlation, create a heatmap
            corr_data = result.get("data")

            visualizations.append(
                {
                    "id": f"correlation_{uuid.uuid4().hex[:6]}",
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data_config": {
                        "z": corr_data.values.tolist(),
                        "x": corr_data.columns.tolist(),
                        "y": corr_data.index.tolist(),
                        "text_values": True,
                    },
                    "visual_config": {"color_scale": "RdBu_r", "symmetric_scale": True},
                    "analysis_id": result.get("title"),
                    "layout_weight": 2,
                }
            )

            # Also add a scatter plot for the top correlated pair
            corr_matrix = corr_data.values
            np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlations

            if corr_matrix.size > 1:
                # Find strongest correlation
                max_idx = np.unravel_index(
                    np.argmax(np.abs(corr_matrix)), corr_matrix.shape
                )
                var1, var2 = (
                    corr_data.columns[max_idx[0]],
                    corr_data.columns[max_idx[1]],
                )
                corr_val = corr_matrix[max_idx]

                visualizations.append(
                    {
                        "id": f"scatter_{uuid.uuid4().hex[:6]}",
                        "type": "scatter_plot",
                        "title": f"Relationship: {var1} vs {var2} (Correlation: {corr_val:.2f})",
                        "data_config": {"x": var1, "y": var2},
                        "visual_config": {"trendline": "ols", "opacity": 0.7},
                        "analysis_id": result.get("title"),
                        "layout_weight": 2,
                    }
                )

        elif analysis_type == "breakdown":
            # For breakdown, create a bar chart
            data = result.get("data")
            dimension = data.columns[0]  # First column is dimension
            metrics = data.columns[1:]  # Rest are metrics

            visualizations.append(
                {
                    "id": f"breakdown_{uuid.uuid4().hex[:6]}",
                    "type": "bar_chart",
                    "title": f"Metrics by {dimension}",
                    "data_config": {
                        "x": dimension,
                        "y": list(metrics),
                        "data": data.to_dict("records"),
                    },
                    "visual_config": {
                        "color_discrete_sequence": px.colors.qualitative.Pastel,
                        "barmode": "group",
                    },
                    "analysis_id": result.get("title"),
                    "layout_weight": 2,
                }
            )

        elif analysis_type == "distribution":
            # For distribution, create histograms
            for metric in result["data"].index:
                visualizations.append(
                    {
                        "id": f"dist_{metric.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}",
                        "type": "histogram",
                        "title": f"Distribution of {metric}",
                        "data_config": {"x": metric},
                        "visual_config": {
                            "nbins": 20,
                            "opacity": 0.7,
                            "show_mean": True,
                        },
                        "analysis_id": result.get("title"),
                        "layout_weight": 2,
                    }
                )

        elif analysis_type == "grouping":
            # For grouping, create a heatmap or grouped bar chart
            data = result.get("data")

            if isinstance(data.columns, pd.MultiIndex):
                # Pivot table - use heatmap
                visualizations.append(
                    {
                        "id": f"grouping_heatmap_{uuid.uuid4().hex[:6]}",
                        "type": "heatmap",
                        "title": "Grouped Analysis",
                        "data_config": {
                            "z": data.values.tolist(),
                            "x": data.columns.tolist(),
                            "y": data.index.tolist(),
                            "text_values": True,
                        },
                        "visual_config": {"color_scale": "Viridis"},
                        "analysis_id": result.get("title"),
                        "layout_weight": 3,
                    }
                )
            else:
                # Regular groupby - use bar chart
                dimension = data.columns[0]  # First column is dimension
                metrics = data.columns[1:]  # Rest are metrics

                visualizations.append(
                    {
                        "id": f"grouping_bar_{uuid.uuid4().hex[:6]}",
                        "type": "bar_chart",
                        "title": "Grouped Analysis",
                        "data_config": {
                            "x": dimension,
                            "y": list(metrics),
                            "data": data.to_dict("records"),
                        },
                        "visual_config": {
                            "color_discrete_sequence": px.colors.qualitative.Bold,
                            "barmode": "group",
                        },
                        "analysis_id": result.get("title"),
                        "layout_weight": 3,
                    }
                )

        elif analysis_type == "pareto":
            # For Pareto, create a combined bar and line chart
            data = result.get("data")
            dimension = data.columns[0]
            metric = data.columns[1]

            visualizations.append(
                {
                    "id": f"pareto_{uuid.uuid4().hex[:6]}",
                    "type": "pareto_chart",
                    "title": "Pareto Analysis",
                    "data_config": {
                        "x": dimension,
                        "y": metric,
                        "cumulative": "cumulative_pct",
                        "data": data.to_dict("records"),
                    },
                    "visual_config": {
                        "color_bar": "royalblue",
                        "color_line": "firebrick",
                    },
                    "analysis_id": result.get("title"),
                    "layout_weight": 3,
                }
            )

    return visualizations


def create_visualization(df, viz_config):
    """
    Create a Plotly visualization based on configuration.

    Args:
        df (pandas.DataFrame): Dataset
        viz_config (dict): Visualization configuration

    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    viz_type = viz_config.get("type")
    title = viz_config.get("title", "")
    data_config = viz_config.get("data_config", {})
    visual_config = viz_config.get("visual_config", {})

    fig = None

    # Create different visualization types
    if viz_type == "kpi_card":
        # Special handling for KPI cards - create a simple figure with text
        metric = data_config.get("metric", "")
        value = data_config.get("value", 0)

        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={"text": metric},
                domain={"x": [0, 1], "y": [0, 1]},
            )
        )

        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))

    elif viz_type == "line_chart":
        # Get data - either from custom data or from dataframe
        if "data" in data_config:
            chart_data = pd.DataFrame(data_config["data"])
        else:
            chart_data = df

        x = data_config.get("x")
        y = data_config.get("y", [])

        if not isinstance(y, list):
            y = [y]

        fig = px.line(chart_data, x=x, y=y, title=title)

        # Apply visual configurations
        if visual_config.get("markers", False):
            fig.update_traces(mode="lines+markers")

        if "color_discrete_sequence" in visual_config:
            fig.update_layout(colorway=visual_config["color_discrete_sequence"])

    elif viz_type == "bar_chart":
        # Get data - either from custom data or from dataframe
        if "data" in data_config:
            chart_data = pd.DataFrame(data_config["data"])
        else:
            chart_data = df

        x = data_config.get("x")
        y = data_config.get("y", [])

        if not isinstance(y, list):
            y = [y]

        fig = px.bar(
            chart_data,
            x=x,
            y=y,
            title=title,
            barmode=visual_config.get("barmode", "group"),
        )

        # Apply visual configurations
        if "color_discrete_sequence" in visual_config:
            fig.update_layout(colorway=visual_config["color_discrete_sequence"])

    elif viz_type == "scatter_plot":
        x = data_config.get("x")
        y = data_config.get("y")

        fig = px.scatter(
            df, x=x, y=y, title=title, opacity=visual_config.get("opacity", 0.7)
        )

        # Add trendline if specified
        if visual_config.get("trendline"):
            fig = px.scatter(
                df,
                x=x,
                y=y,
                title=title,
                trendline=visual_config["trendline"],
                opacity=visual_config.get("opacity", 0.7),
            )

    elif viz_type == "histogram":
        x = data_config.get("x")

        fig = px.histogram(
            df,
            x=x,
            title=title,
            nbins=visual_config.get("nbins", 20),
            opacity=visual_config.get("opacity", 0.7),
        )

        # Add mean line if specified
        if visual_config.get("show_mean", False):
            mean_val = df[x].mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right",
            )

    elif viz_type == "heatmap":
        # For heatmap, we typically need the pre-computed z values
        z = data_config.get("z", [])
        x = data_config.get("x", [])
        y = data_config.get("y", [])

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=visual_config.get("color_scale", "Viridis"),
                zauto=not visual_config.get("symmetric_scale", False),
                zmid=0 if visual_config.get("symmetric_scale", False) else None,
                text=z if data_config.get("text_values", False) else None,
                texttemplate="%{text:.2f}"
                if data_config.get("text_values", False)
                else None,
                hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(title=title)

    elif viz_type == "pareto_chart":
        # Get data - either from custom data or from dataframe
        if "data" in data_config:
            chart_data = pd.DataFrame(data_config["data"])
        else:
            chart_data = df

        x = data_config.get("x")
        y = data_config.get("y")
        cumulative = data_config.get("cumulative")

        # Create figure with secondary y-axis
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=chart_data[x],
                y=chart_data[y],
                name=y,
                marker_color=visual_config.get("color_bar", "royalblue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_data[x],
                y=chart_data[cumulative],
                name="Cumulative %",
                marker_color=visual_config.get("color_line", "firebrick"),
                yaxis="y2",
            )
        )

        # Set figure layout
        fig.update_layout(
            title=title,
            yaxis=dict(title=y),
            yaxis2=dict(
                title="Cumulative %", overlaying="y", side="right", range=[0, 100]
            ),
            legend=dict(x=0.01, y=0.99),
        )

    elif viz_type == "pie_chart":
        values = data_config.get("values")
        names = data_config.get("names")

        fig = px.pie(df, values=values, names=names, title=title)

        # Apply visual configurations
        if "hole" in visual_config:
            fig.update_traces(hole=visual_config["hole"])

    # Apply common layout settings
    if fig:
        fig.update_layout(margin=dict(l=10, r=10, b=10, t=40), template="plotly_white")

    return fig


def render_visualization_interface(df, analysis_results, business_context):
    """
    Render the visualization generation interface.

    Args:
        df (pandas.DataFrame): Dataset
        analysis_results (list): Results from analyses
        business_context (dict): Business context information

    Returns:
        list: Generated visualizations
    """
    st.header("Visualization Generation")

    # Generate visualization configs if not already in session
    if "visualization_configs" not in st.session_state:
        st.session_state.visualization_configs = generate_visualization_config(
            analysis_results, business_context
        )

    if "selected_visualizations" not in st.session_state:
        st.session_state.selected_visualizations = []

    # Display tabs for different views
    tab1, tab2 = st.tabs(["Suggested Visualizations", "Selected Visualizations"])

    with tab1:
        st.write("Review and select from the suggested visualizations:")

        if not st.session_state.visualization_configs:
            st.info(
                "No visualizations could be generated from the analyses. Try running more analyses."
            )

        # Display visualization suggestions
        for i, viz_config in enumerate(st.session_state.visualization_configs):
            # Skip if already selected
            if any(
                v["id"] == viz_config["id"]
                for v in st.session_state.selected_visualizations
            ):
                continue

            # Create columns for viz and selection button
            col1, col2 = st.columns([4, 1])

            with col1:
                # Create visualization
                fig = create_visualization(df, viz_config)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(
                        f"**Based on**: {viz_config.get('analysis_id', 'Data analysis')}"
                    )

            with col2:
                # Add selection buttons
                if st.button("Select", key=f"select_viz_{i}", use_container_width=True):
                    st.session_state.selected_visualizations.append(viz_config)
                    st.rerun()

    with tab2:
        if not st.session_state.selected_visualizations:
            st.info(
                "No visualizations selected yet. Go to the Suggested Visualizations tab to select some."
            )
        else:
            st.write(
                f"You have selected {len(st.session_state.selected_visualizations)} visualizations:"
            )

            # Display selected visualizations
            for i, viz_config in enumerate(st.session_state.selected_visualizations):
                col1, col2 = st.columns([4, 1])

                with col1:
                    # Create visualization
                    fig = create_visualization(df, viz_config)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Allow customization (title only for now)
                    new_title = st.text_input(
                        "Title", viz_config.get("title", ""), key=f"title_{i}"
                    )
                    if new_title != viz_config.get("title", ""):
                        # Update title
                        st.session_state.selected_visualizations[i]["title"] = new_title
                        st.rerun()

                    # Remove button
                    if st.button(
                        "Remove", key=f"remove_viz_{i}", use_container_width=True
                    ):
                        st.session_state.selected_visualizations.pop(i)
                        st.rerun()

                st.markdown("---")

    # Continue button
    if st.session_state.selected_visualizations:
        if st.button("Continue to Dashboard Assembly"):
            return st.session_state.selected_visualizations

    return None
