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
            if isinstance(result["data"], pd.DataFrame):
                metrics = [
                    col
                    for col in result["data"].index
                    if col in business_context.get("key_metrics", [])
                ]

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
            if (
                data is not None and len(data.columns) > 1
            ):  # Need at least time + one metric
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
            if corr_data is not None and not corr_data.empty:
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
                        "visual_config": {
                            "color_scale": "RdBu_r",
                            "symmetric_scale": True,
                        },
                        "analysis_id": result.get("title"),
                        "layout_weight": 2,
                    }
                )

                # Also add a scatter plot for the top correlated pair
                if corr_data.size > 1:
                    # Find strongest correlation
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

                        # Check if both columns exist in the dataset
                        if var1 in business_context.get(
                            "key_metrics", []
                        ) and var2 in business_context.get("key_metrics", []):
                            visualizations.append(
                                {
                                    "id": f"scatter_{uuid.uuid4().hex[:6]}",
                                    "type": "scatter_plot",
                                    "title": f"Relationship: {var1} vs {var2} (Correlation: {corr_val:.2f})",
                                    "data_config": {"x": var1, "y": var2},
                                    "visual_config": {
                                        "trendline": "ols",
                                        "opacity": 0.7,
                                    },
                                    "analysis_id": result.get("title"),
                                    "layout_weight": 2,
                                }
                            )

        elif analysis_type == "breakdown":
            # For breakdown, create a bar chart
            data = result.get("data")
            if data is not None and not data.empty and len(data.columns) > 1:
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
            # For distribution, only create histograms for actual dataset columns
            # Get actual metrics from business context that exist in dataset
            valid_metrics = [
                m
                for m in business_context.get("key_metrics", [])
                if m in result.get("data", pd.DataFrame()).index
            ]

            for metric in valid_metrics:
                if metric in result.get("data", pd.DataFrame()).index:
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

            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    # Pivot table - use heatmap
                    visualizations.append(
                        {
                            "id": f"grouping_heatmap_{uuid.uuid4().hex[:6]}",
                            "type": "heatmap",
                            "title": "Grouped Analysis",
                            "data_config": {
                                "z": data.values.tolist(),
                                "x": [str(x) for x in data.columns.tolist()],
                                "y": [str(y) for y in data.index.tolist()],
                                "text_values": True,
                            },
                            "visual_config": {"color_scale": "Viridis"},
                            "analysis_id": result.get("title"),
                            "layout_weight": 3,
                        }
                    )
                elif (
                    len(data.columns) > 1
                ):  # Need at least one dimension and one metric
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
            if (
                data is not None and not data.empty and len(data.columns) > 2
            ):  # Need dimension, metric, and cumulative
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

    # Validate columns exist in dataframe
    def validate_columns(cols):
        if isinstance(cols, list):
            return [col for col in cols if col in df.columns]
        elif cols in df.columns:
            return cols
        else:
            return None

    fig = None

    try:
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

            x = validate_columns(data_config.get("x"))
            y = validate_columns(data_config.get("y", []))

            # Skip if invalid columns
            if x is None or (isinstance(y, list) and not y):
                return None

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

            x = validate_columns(data_config.get("x"))
            y = validate_columns(data_config.get("y", []))

            # Skip if invalid columns
            if x is None or (isinstance(y, list) and not y):
                return None

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
            x = validate_columns(data_config.get("x"))
            y = validate_columns(data_config.get("y"))

            # Skip if invalid columns
            if x is None or y is None:
                return None

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
            x = validate_columns(data_config.get("x"))

            # Skip if invalid column
            if x is None:
                return None

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

            # Validate x and y if they're column names
            if isinstance(x, str):
                x = validate_columns(x)
                if x is None:
                    return None

            if isinstance(y, str):
                y = validate_columns(y)
                if y is None:
                    return None

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

            x = validate_columns(data_config.get("x"))
            y = validate_columns(data_config.get("y"))
            cumulative = data_config.get("cumulative")

            # Skip if invalid columns
            if x is None or y is None:
                return None

            # Check if cumulative column exists
            if cumulative and cumulative not in chart_data.columns:
                # Calculate cumulative percentages
                chart_data = chart_data.sort_values(by=y, ascending=False)
                chart_data["cumulative"] = chart_data[y].cumsum()
                chart_data["cumulative_pct"] = (
                    chart_data["cumulative"] / chart_data[y].sum() * 100
                )
                cumulative = "cumulative_pct"

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
            values = validate_columns(data_config.get("values"))
            names = validate_columns(data_config.get("names"))

            # Skip if invalid columns
            if values is None or names is None:
                return None

            fig = px.pie(df, values=values, names=names, title=title)

            # Apply visual configurations
            if "hole" in visual_config:
                fig.update_traces(hole=visual_config["hole"])

        # Apply common layout settings
        if fig:
            fig.update_layout(
                margin=dict(l=10, r=10, b=10, t=40), template="plotly_white"
            )

        return fig

    except Exception as e:
        st.warning(f"Error creating visualization: {str(e)}")
        return None


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

    # Check if we have analysis results to work with
    if not analysis_results:
        st.error(
            "No analysis results available. Please complete the analysis step first."
        )
        if st.button("Go to Advanced Analysis"):
            st.session_state.step = "advanced_analysis"
            st.rerun()
        return None

    # Generate some basic visualizations if none exist yet
    if (
        "visualization_configs" not in st.session_state
        or not st.session_state.visualization_configs
    ):
        st.info("Generating visualization suggestions...")

        # Ensure numpy is imported
        import numpy as np

        # Create basic visualizations based on business context
        basic_visualizations = []
        metrics = business_context.get("key_metrics", [])
        dimensions = business_context.get("dimensions", [])
        time_dimension = business_context.get("time_dimension")

        # Add a time series chart if time dimension exists
        if time_dimension and metrics:
            for metric in metrics[:3]:  # Top 3 metrics
                viz_id = f"timeseries_{metric}_{uuid.uuid4().hex[:6]}"
                basic_visualizations.append(
                    {
                        "id": viz_id,
                        "type": "line_chart",
                        "title": f"{metric} Over Time",
                        "data_config": {"x": time_dimension, "y": metric},
                        "visual_config": {
                            "markers": True,
                            "color_discrete_sequence": px.colors.qualitative.G10,
                        },
                        "analysis_id": "Basic Time Series",
                        "layout_weight": 2,
                    }
                )

        # Add breakdown by dimension if dimensions exist
        if dimensions and metrics:
            for dim in dimensions[:2]:  # Top 2 dimensions
                for metric in metrics[:2]:  # Top 2 metrics
                    viz_id = f"breakdown_{dim}_{metric}_{uuid.uuid4().hex[:6]}"
                    basic_visualizations.append(
                        {
                            "id": viz_id,
                            "type": "bar_chart",
                            "title": f"{metric} by {dim}",
                            "data_config": {"x": dim, "y": metric},
                            "visual_config": {
                                "color_discrete_sequence": px.colors.qualitative.Pastel
                            },
                            "analysis_id": "Basic Breakdown",
                            "layout_weight": 2,
                        }
                    )

        # Add correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr().round(2).values.tolist()

            viz_id = f"correlation_{uuid.uuid4().hex[:6]}"
            basic_visualizations.append(
                {
                    "id": viz_id,
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data_config": {
                        "z": corr_matrix,
                        "x": numeric_cols,
                        "y": numeric_cols,
                        "text_values": True,
                    },
                    "visual_config": {"color_scale": "RdBu_r", "symmetric_scale": True},
                    "analysis_id": "Basic Correlation",
                    "layout_weight": 2,
                }
            )

        # Add to existing visualization configs
        existing_configs = st.session_state.get("visualization_configs", [])
        st.session_state.visualization_configs = existing_configs + basic_visualizations

        # If there are analysis results, also add those visualizations
        if analysis_results:
            analysis_visualizations = generate_visualization_config(
                analysis_results, business_context
            )
            st.session_state.visualization_configs.extend(analysis_visualizations)

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
                try:
                    fig = create_visualization(df, viz_config)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback for visualization that couldn't be created
                        st.warning(
                            f"Unable to create visualization: {viz_config.get('title', 'Untitled')}"
                        )
                        st.info(
                            f"Based on: {viz_config.get('analysis_id', 'Analysis')}"
                        )
                except Exception as e:
                    # Handle any errors in visualization creation
                    st.warning(
                        f"Error creating visualization: {viz_config.get('title', 'Untitled')}"
                    )
                    st.info(f"Based on: {viz_config.get('analysis_id', 'Analysis')}")
                    st.error(f"Error details: {str(e)}")

                # Always show what analysis this is based on
                st.write(
                    f"**Based on**: {viz_config.get('analysis_id', 'Data analysis')}"
                )

            with col2:
                # Add selection buttons (always show these even if visualization failed)
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
