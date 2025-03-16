import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def suggest_analyses(df, business_context):
    """
    Suggest analyses based on the dataset and business context.

    Args:
        df (pandas.DataFrame): Dataset
        business_context (dict): Business context information

    Returns:
        list: Suggested analyses
    """
    suggestions = []

    # Get context information
    domain = business_context.get("domain", "unknown")
    metrics = business_context.get("key_metrics", [])
    dimensions = business_context.get("dimensions", [])
    time_dimension = business_context.get("time_dimension")

    # Basic descriptive statistics for all metrics
    if metrics:
        suggestions.append(
            {
                "type": "descriptive",
                "title": "Descriptive Statistics",
                "description": "Basic statistical measures for key metrics",
                "metrics": metrics,
                "dimensions": [],
            }
        )

    # Trend analysis if time dimension exists
    if time_dimension and metrics:
        suggestions.append(
            {
                "type": "time_series",
                "title": "Time Series Analysis",
                "description": f"Analyze trends in key metrics over time",
                "metrics": metrics[:3],  # Limit to top 3 metrics
                "dimensions": [time_dimension],
                "parameters": {"frequency": "auto"},
            }
        )

    # Breakdowns by dimensions
    if dimensions and metrics:
        for dim in dimensions[:2]:  # Limit to top 2 dimensions
            suggestions.append(
                {
                    "type": "breakdown",
                    "title": f"Breakdown by {dim}",
                    "description": f"Compare metrics across different {dim} values",
                    "metrics": metrics[:2],  # Limit to top 2 metrics
                    "dimensions": [dim],
                }
            )

    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) >= 2:
        suggestions.append(
            {
                "type": "correlation",
                "title": "Correlation Analysis",
                "description": "Identify relationships between numeric variables",
                "metrics": list(numeric_cols)[:5],  # Limit to top 5 metrics
                "dimensions": [],
            }
        )

    # Distribution analysis
    if metrics:
        suggestions.append(
            {
                "type": "distribution",
                "title": "Distribution Analysis",
                "description": "Analyze the distribution of key metrics",
                "metrics": metrics[:3],  # Limit to top 3 metrics
                "dimensions": [],
            }
        )

    # Domain-specific analyses
    if domain == "sales":
        if "product" in df.columns.str.lower() and "revenue" in df.columns.str.lower():
            suggestions.append(
                {
                    "type": "pareto",
                    "title": "Product Pareto Analysis",
                    "description": "Identify which products contribute most to revenue",
                    "metrics": ["revenue"],
                    "dimensions": ["product"],
                }
            )

    elif domain == "marketing":
        if (
            "campaign" in df.columns.str.lower()
            and "conversion" in df.columns.str.lower()
        ):
            suggestions.append(
                {
                    "type": "comparison",
                    "title": "Campaign Effectiveness",
                    "description": "Compare conversion rates across different campaigns",
                    "metrics": ["conversion"],
                    "dimensions": ["campaign"],
                }
            )

    # Add grouping analysis if dimensions exist
    if dimensions and metrics:
        suggestions.append(
            {
                "type": "grouping",
                "title": "Grouped Metric Analysis",
                "description": f"Analyze metrics grouped by dimensions",
                "metrics": metrics[:2],
                "dimensions": dimensions[:2],
                "parameters": {"aggregation": "mean"},
            }
        )

    return suggestions


def run_analysis(df, analysis_config):
    """
    Run a specific analysis based on configuration.

    Args:
        df (pandas.DataFrame): Dataset
        analysis_config (dict): Analysis configuration

    Returns:
        dict: Analysis results
    """
    analysis_type = analysis_config.get("type")
    metrics = analysis_config.get("metrics", [])
    dimensions = analysis_config.get("dimensions", [])
    parameters = analysis_config.get("parameters", {})

    # Results structure
    results = {
        "type": analysis_type,
        "title": analysis_config.get("title", "Analysis"),
        "description": analysis_config.get("description", ""),
        "data": None,
        "summary": "",
        "visualization": None,
    }

    # Run different analyses based on type
    if analysis_type == "descriptive":
        # Basic descriptive statistics
        stats_df = df[metrics].describe().T
        stats_df["missing"] = df[metrics].isna().sum()
        stats_df["missing_pct"] = (df[metrics].isna().sum() / len(df) * 100).round(2)

        results["data"] = stats_df
        results["summary"] = (
            "Descriptive statistics provide an overview of central tendency and dispersion of your key metrics."
        )

        # Create visualization - boxplot
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Box(y=df[metric], name=metric))

        fig.update_layout(
            title="Distribution of Key Metrics", yaxis_title="Value", showlegend=False
        )

        results["visualization"] = fig

    elif analysis_type == "time_series":
        # Ensure time dimension is datetime
        time_col = dimensions[0]
        if df[time_col].dtype != "datetime64[ns]":
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                results["summary"] = (
                    f"Error: Could not convert {time_col} to datetime format."
                )
                return results

        # Group by time dimension and calculate metrics
        frequency = parameters.get("frequency", "auto")

        if frequency == "auto":
            # Try to determine appropriate frequency
            date_range = (df[time_col].max() - df[time_col].min()).days
            if date_range > 365 * 2:
                frequency = "Y"  # Yearly
            elif date_range > 90:
                frequency = "M"  # Monthly
            elif date_range > 21:
                frequency = "W"  # Weekly
            else:
                frequency = "D"  # Daily

        # Create a datetime index
        df_time = df.set_index(time_col)

        # Resample based on frequency
        time_data = {}
        for metric in metrics:
            resampled = df_time[metric].resample(frequency).mean()
            time_data[metric] = resampled

        time_df = pd.DataFrame(time_data)
        time_df = time_df.reset_index()

        results["data"] = time_df

        # Calculate trends
        trend_analysis = []
        for metric in metrics:
            values = time_df[metric].dropna()
            if len(values) > 1:
                # Simple linear regression for trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, values
                )

                trend_direction = "increasing" if slope > 0 else "decreasing"
                trend_strength = abs(r_value)

                trend_analysis.append(
                    f"{metric} shows a {trend_direction} trend (correlation: {trend_strength:.2f})"
                )

        results["summary"] = (
            "Time series analysis shows how metrics change over time. "
            + " ".join(trend_analysis)
        )

        # Create visualization - line chart
        fig = px.line(
            time_df,
            x=time_col,
            y=metrics,
            title="Metrics Over Time",
            labels={"value": "Value", "variable": "Metric"},
        )

        results["visualization"] = fig

    elif analysis_type == "correlation":
        # Correlation analysis
        corr_df = df[metrics].corr().round(2)

        results["data"] = corr_df

        # Find strongest correlations
        corr_matrix = corr_df.values
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlations

        # Get indices of top 3 correlations
        flat_corr = corr_matrix.flatten()
        top_indices = np.argsort(np.abs(flat_corr))[-3:]

        top_correlations = []
        for idx in top_indices:
            if flat_corr[idx] != 0:  # Skip if correlation is 0
                i, j = idx // len(metrics), idx % len(metrics)
                metric1, metric2 = metrics[i], metrics[j]
                corr_val = flat_corr[idx]
                relationship = "positive" if corr_val > 0 else "negative"
                strength = (
                    "strong"
                    if abs(corr_val) > 0.7
                    else "moderate"
                    if abs(corr_val) > 0.3
                    else "weak"
                )

                top_correlations.append(
                    f"{metric1} and {metric2} have a {strength} {relationship} correlation ({corr_val:.2f})"
                )

        if top_correlations:
            results["summary"] = (
                "Correlation analysis reveals relationships between metrics. "
                + " ".join(top_correlations)
            )
        else:
            results["summary"] = (
                "No significant correlations found between the selected metrics."
            )

        # Create visualization - heatmap
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
        )

        results["visualization"] = fig

    elif analysis_type == "breakdown":
        # Breakdown by dimension
        dim = dimensions[0]

        # Get top categories by count
        top_categories = df[dim].value_counts().head(10).index.tolist()

        # Filter to top categories
        breakdown_df = df[df[dim].isin(top_categories)]

        # Group by dimension and calculate metrics
        grouped = breakdown_df.groupby(dim)[metrics].mean().reset_index()

        results["data"] = grouped

        # Identify highest and lowest categories
        insights = []
        for metric in metrics:
            max_category = grouped.loc[grouped[metric].idxmax(), dim]
            min_category = grouped.loc[grouped[metric].idxmin(), dim]

            insights.append(
                f"{max_category} has the highest average {metric}, while {min_category} has the lowest."
            )

        results["summary"] = (
            f"Breakdown analysis by {dim} shows how metrics vary across categories. "
            + " ".join(insights)
        )

        # Create visualization - bar chart
        fig = px.bar(
            grouped, x=dim, y=metrics, title=f"Metrics by {dim}", barmode="group"
        )

        results["visualization"] = fig

    elif analysis_type == "distribution":
        # Distribution analysis
        dist_summary = []

        for metric in metrics:
            # Basic distribution stats
            mean_val = df[metric].mean()
            median_val = df[metric].median()
            skew_val = df[metric].skew()

            # Determine distribution shape
            if abs(skew_val) < 0.5:
                shape = "symmetrical"
            elif skew_val > 0:
                shape = "right-skewed"
            else:
                shape = "left-skewed"

            dist_summary.append(
                f"{metric} has a {shape} distribution with mean {mean_val:.2f} and median {median_val:.2f}."
            )

        results["data"] = df[metrics].describe()
        results["summary"] = (
            "Distribution analysis examines the spread and shape of your metrics. "
            + " ".join(dist_summary)
        )

        # Create visualization - histogram
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Histogram(x=df[metric], name=metric, opacity=0.7))

        fig.update_layout(
            barmode="overlay",
            title="Distribution of Key Metrics",
            xaxis_title="Value",
            yaxis_title="Count",
        )

        results["visualization"] = fig

    elif analysis_type == "grouping":
        # Grouped analysis
        dim1 = dimensions[0]
        dim2 = dimensions[1] if len(dimensions) > 1 else None

        if dim2:
            # Two dimensions - create pivot table
            for metric in metrics:
                pivot_df = pd.pivot_table(
                    df,
                    values=metric,
                    index=dim1,
                    columns=dim2,
                    aggfunc=parameters.get("aggregation", "mean"),
                )

                results["data"] = pivot_df

                # Create visualization - heatmap
                fig = px.imshow(
                    pivot_df, text_auto=True, title=f"{metric} by {dim1} and {dim2}"
                )

                results["visualization"] = fig
                break  # Only use first metric for visualization
        else:
            # One dimension - simple groupby
            grouped = (
                df.groupby(dim1)[metrics]
                .agg(parameters.get("aggregation", "mean"))
                .reset_index()
            )
            results["data"] = grouped

            # Create visualization - bar chart
            fig = px.bar(
                grouped, x=dim1, y=metrics, title=f"Metrics by {dim1}", barmode="group"
            )

            results["visualization"] = fig

        results["summary"] = (
            f"Grouping analysis shows how metrics vary across different combinations of dimensions."
        )

    elif analysis_type == "pareto":
        # Pareto analysis
        pareto_dim = dimensions[0]
        pareto_metric = metrics[0]

        # Group by dimension and sum metric
        grouped = df.groupby(pareto_dim)[pareto_metric].sum().reset_index()

        # Sort by metric descending
        grouped = grouped.sort_values(pareto_metric, ascending=False)

        # Calculate cumulative percentage
        grouped["cumulative"] = grouped[pareto_metric].cumsum()
        grouped["cumulative_pct"] = (
            grouped["cumulative"] / grouped[pareto_metric].sum() * 100
        )

        results["data"] = grouped

        # Get number of items for 80% of total
        items_80pct = (grouped["cumulative_pct"] <= 80).sum()
        total_items = len(grouped)

        results["summary"] = (
            f"Pareto analysis shows that {items_80pct} out of {total_items} {pareto_dim} items account for 80% of total {pareto_metric}."
        )

        # Create visualization - pareto chart
        fig = go.Figure()

        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=grouped[pareto_dim][:10],  # Top 10 items
                y=grouped[pareto_metric][:10],
                name=pareto_metric,
            )
        )

        # Add line chart for cumulative percentage
        fig.add_trace(
            go.Scatter(
                x=grouped[pareto_dim][:10],
                y=grouped["cumulative_pct"][:10],
                name="Cumulative %",
                yaxis="y2",
            )
        )

        # Layout with dual y-axis
        fig.update_layout(
            title=f"Pareto Analysis: {pareto_metric} by {pareto_dim}",
            yaxis=dict(title=pareto_metric),
            yaxis2=dict(
                title="Cumulative %", overlaying="y", side="right", range=[0, 100]
            ),
        )

        results["visualization"] = fig

    return results


def render_analysis_interface(df, business_context):
    """
    Render the advanced analysis interface.

    Args:
        df (pandas.DataFrame): Dataset
        business_context (dict): Business context information

    Returns:
        list: Completed analyses
    """
    st.header("Advanced Analysis")

    # Get suggested analyses
    suggested_analyses = suggest_analyses(df, business_context)

    st.write("Select analyses to perform on your data.")

    # Store selected analyses
    if "selected_analyses" not in st.session_state:
        st.session_state.selected_analyses = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []

    # Display suggested analyses with checkboxes
    for i, analysis in enumerate(suggested_analyses):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(analysis["title"])
            st.write(analysis["description"])
            st.write(f"Metrics: {', '.join(analysis['metrics'])}")
            st.write(
                f"Dimensions: {', '.join(analysis['dimensions']) if analysis['dimensions'] else 'None'}"
            )

        with col2:
            # Check if already selected
            is_selected = any(
                a["title"] == analysis["title"]
                for a in st.session_state.selected_analyses
            )

            if st.button(
                "Select" if not is_selected else "Selected ✓",
                key=f"select_{i}",
                disabled=is_selected,
                use_container_width=True,
            ):
                st.session_state.selected_analyses.append(analysis)
                st.rerun()

        st.markdown("---")

    # Run selected analyses
    if st.session_state.selected_analyses:
        st.header("Selected Analyses")

        for i, analysis in enumerate(st.session_state.selected_analyses):
            # Check if analysis has been run already
            existing_result = next(
                (
                    r
                    for r in st.session_state.analysis_results
                    if r["title"] == analysis["title"]
                ),
                None,
            )

            with st.expander(f"{analysis['title']}", expanded=existing_result is None):
                if existing_result:
                    # Show existing results
                    st.write(existing_result["description"])
                    st.write(existing_result["summary"])

                    if existing_result["visualization"]:
                        st.plotly_chart(
                            existing_result["visualization"], use_container_width=True
                        )

                    if isinstance(existing_result["data"], pd.DataFrame):
                        st.dataframe(existing_result["data"])
                else:
                    # Run new analysis
                    with st.spinner(f"Running {analysis['title']}..."):
                        result = run_analysis(df, analysis)
                        st.session_state.analysis_results.append(result)

                        st.write(result["description"])
                        st.write(result["summary"])

                        if result["visualization"]:
                            st.plotly_chart(
                                result["visualization"], use_container_width=True
                            )

                        if isinstance(result["data"], pd.DataFrame):
                            st.dataframe(result["data"])

    # Continue button
    if st.session_state.analysis_results:
        if st.button("Continue to Visualization Generation"):
            return st.session_state.analysis_results

    return None
