import streamlit as st
import pandas as pd
from utils.openai_utils import get_chat_completion
import json


def detect_business_domain(df, profile):
    """
    Attempt to detect the business domain based on data analysis.

    Args:
        df (pandas.DataFrame): Dataset
        profile (dict): Data profiling results

    Returns:
        dict: Business context information
    """
    # Get column names and sample data
    columns = list(df.columns)
    sample_data = df.head(5).to_dict(orient="records")

    # Try to infer business domain from AI analysis if available
    ai_analysis = profile.get("ai_analysis", {})
    dataset_purpose = (
        ai_analysis.get("dataset_purpose", "") if isinstance(ai_analysis, dict) else ""
    )

    # Initialize with empty context
    context = {
        "domain": "unknown",
        "potential_domains": [
            "sales",
            "marketing",
            "operations",
            "finance",
            "hr",
            "customer_service",
            "other",
        ],
        "key_metrics": [],
        "dimensions": [],
        "time_columns": [],
        "entity_columns": [],
    }

    # Try to identify domain-specific keywords in column names
    domains_keywords = {
        "sales": [
            "sale",
            "revenue",
            "customer",
            "product",
            "order",
            "transaction",
            "price",
            "discount",
        ],
        "marketing": [
            "campaign",
            "cost",
            "conversion",
            "click",
            "impression",
            "lead",
            "channel",
            "ad",
            "segment",
        ],
        "operations": [
            "inventory",
            "supply",
            "demand",
            "logistics",
            "shipping",
            "warehouse",
            "production",
            "capacity",
        ],
        "finance": [
            "profit",
            "loss",
            "expense",
            "budget",
            "cost",
            "revenue",
            "account",
            "balance",
            "payment",
        ],
        "hr": [
            "employee",
            "salary",
            "performance",
            "department",
            "hire",
            "attrition",
            "training",
            "recruitment",
        ],
        "customer_service": [
            "ticket",
            "issue",
            "resolution",
            "satisfaction",
            "support",
            "feedback",
            "complaint",
            "response",
        ],
    }

    # Count domain-related keywords
    domain_scores = {domain: 0 for domain in domains_keywords}
    for col in columns:
        col_lower = col.lower()
        for domain, keywords in domains_keywords.items():
            for keyword in keywords:
                if keyword in col_lower:
                    domain_scores[domain] += 1

    # Get the highest scoring domain
    if domain_scores:
        max_score = max(domain_scores.values())
        if max_score > 0:
            context["domain"] = max(domain_scores, key=domain_scores.get)

    # If dataset_purpose contains domain info, use that instead
    for domain in domains_keywords:
        if domain in dataset_purpose.lower():
            context["domain"] = domain
            break

    # Identify potential metrics (numeric columns)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    context["key_metrics"] = list(numeric_cols)

    # Identify dimensions (categorical columns)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    context["dimensions"] = list(categorical_cols)

    # Identify time columns from profile
    date_candidates = profile.get("date_candidates", [])
    context["time_columns"] = date_candidates

    # Try to identify entity columns (IDs, names)
    entity_keywords = ["id", "name", "code", "key", "number"]
    entity_cols = []
    for col in columns:
        col_lower = col.lower()
        for keyword in entity_keywords:
            if keyword in col_lower:
                entity_cols.append(col)
                break
    context["entity_columns"] = entity_cols

    return context


def render_business_understanding(df, profile):
    """
    Render the business understanding interface.

    Args:
        df (pandas.DataFrame): Dataset
        profile (dict): Data profiling results

    Returns:
        dict: Business context defined by user
    """
    st.header("Business Understanding")

    # Detect business domain
    auto_context = detect_business_domain(df, profile)

    st.write(
        "Let's establish the business context of your data to create meaningful dashboards."
    )

    # Business domain selection
    st.subheader("Business Domain")
    domain_options = [
        "Sales",
        "Marketing",
        "Operations",
        "Finance",
        "HR",
        "Customer Service",
        "Other",
    ]

    # Get index of detected domain
    detected_index = 0
    for i, option in enumerate(domain_options):
        if option.lower() == auto_context["domain"]:
            detected_index = i
            break

    domain = st.selectbox(
        "Select the primary business domain",
        domain_options,
        index=detected_index,
        help="This helps contextualize your data in business terms",
    )

    if domain == "Other":
        custom_domain = st.text_input("Specify domain")
        if custom_domain:
            domain = custom_domain

    # Business goals
    st.subheader("Business Goals")
    st.write("What business questions are you trying to answer with this data?")

    # Default goals based on domain
    default_goals = {
        "Sales": [
            "Track sales performance over time",
            "Identify top-performing products",
            "Analyze regional sales trends",
        ],
        "Marketing": [
            "Evaluate campaign effectiveness",
            "Track conversion rates",
            "Analyze customer segmentation",
        ],
        "Operations": [
            "Monitor inventory levels",
            "Optimize supply chain",
            "Track production efficiency",
        ],
        "Finance": [
            "Analyze profit and loss",
            "Track expenses by category",
            "Monitor budget variance",
        ],
        "HR": [
            "Track employee performance",
            "Analyze attrition rates",
            "Monitor department metrics",
        ],
        "Customer Service": [
            "Track ticket resolution time",
            "Monitor customer satisfaction",
            "Analyze common issues",
        ],
    }

    goals_options = default_goals.get(
        domain,
        [
            "Explore data relationships",
            "Identify patterns",
            "Track performance metrics",
        ],
    )

    goals = []
    for i, goal in enumerate(goals_options):
        if st.checkbox(goal, key=f"goal_{i}", value=True):
            goals.append(goal)

    custom_goal = st.text_input("Add a custom business goal (optional)")
    if custom_goal:
        goals.append(custom_goal)

    # Key metrics
    st.subheader("Key Metrics")
    st.write("Select the primary metrics you want to analyze.")

    metric_cols = auto_context["key_metrics"]
    selected_metrics = []

    # Create columns for better layout
    metric_cols_layout = st.columns(2)

    for i, metric in enumerate(metric_cols):
        col_idx = i % 2
        with metric_cols_layout[col_idx]:
            if st.checkbox(metric, key=f"metric_{i}", value=True):
                selected_metrics.append(metric)

    # Dimensions
    st.subheader("Dimensions")
    st.write("Select the dimensions to slice your data by.")

    dimension_cols = auto_context["dimensions"]
    selected_dimensions = []

    # Create columns for better layout
    dim_cols_layout = st.columns(2)

    for i, dim in enumerate(dimension_cols):
        col_idx = i % 2
        with dim_cols_layout[col_idx]:
            if st.checkbox(dim, key=f"dim_{i}", value=True):
                selected_dimensions.append(dim)

    # Time dimension
    st.subheader("Time Dimension")
    st.write("Identify your primary time dimension for trend analysis.")

    # Get date columns from both profile and actual datetime columns in the DataFrame
    time_cols = auto_context["time_columns"]

    # Also check directly for datetime columns in the DataFrame
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in datetime_cols:
        if col not in time_cols:
            time_cols.append(col)

    time_dimension = None

    if time_cols:
        time_dimension = st.selectbox(
            "Select time dimension",
            ["None"] + time_cols,
            index=1 if time_cols else 0,
            help="Used for time-based analysis",
        )

        if time_dimension == "None":
            time_dimension = None
    else:
        st.info(
            "No date/time columns detected. Time-based analysis may not be available."
        )

    # Entity column
    st.subheader("Primary Entity")
    st.write(
        "Select the column that represents your primary business entity (e.g., customer ID, product ID)."
    )

    entity_cols = auto_context["entity_columns"]
    primary_entity = None

    if entity_cols:
        primary_entity = st.selectbox(
            "Select primary entity",
            ["None"] + entity_cols,
            index=1 if entity_cols else 0,
            help="Used for entity-based analysis",
        )

        if primary_entity == "None":
            primary_entity = None
    else:
        st.info(
            "No entity columns detected. Entity-based analysis may not be available."
        )

    # Submit button
    if st.button("Confirm Business Context"):
        # Construct business context
        business_context = {
            "domain": domain.lower(),
            "goals": goals,
            "key_metrics": selected_metrics,
            "dimensions": selected_dimensions,
            "time_dimension": time_dimension,
            "primary_entity": primary_entity,
        }

        return business_context

    return None
