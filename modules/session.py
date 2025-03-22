from datetime import datetime
import json
import streamlit as st


class SessionState:
    """Manages state persistence across the application."""

    def __init__(self):
        """Initialize a new session state if one doesn't exist."""
        if "initialized" not in st.session_state:
            self._initialize_session()

    def _initialize_session(self):
        """Set up initial session state values."""
        st.session_state.initialized = True
        st.session_state.step = "upload"  # Current step in the workflow
        st.session_state.dataset = None  # The pandas DataFrame
        st.session_state.dataset_name = None  # Original filename
        st.session_state.profile = None  # Data profiling results
        st.session_state.preprocessing = []  # Applied preprocessing steps
        st.session_state.business_context = {}  # Business understanding
        st.session_state.analyses = []  # Performed analyses
        st.session_state.analysis_results = []  # Complete analysis results with visualizations
        st.session_state.visualization_configs = []  # Generated visualization configurations
        st.session_state.selected_visualizations = []  # User-selected visualizations
        st.session_state.visualizations = []  # Generated visualizations
        st.session_state.dashboard_config = {}  # Dashboard layout and settings
        st.session_state.history = []  # Step history for navigation

    def set_dataset(self, df, filename):
        """Store the dataset in session state."""
        st.session_state.dataset = df
        st.session_state.dataset_name = filename
        self._add_history_entry("upload", {"filename": filename, "shape": df.shape})

    def set_profile(self, profile):
        """Store data profiling results."""
        st.session_state.profile = profile
        self._add_history_entry(
            "profile",
            {
                "summary": "Data profiling completed",
                "timestamp": datetime.now().isoformat(),
            },
        )

    def add_preprocessing_step(self, step_info):
        """Add a preprocessing step to history."""
        st.session_state.preprocessing.append(step_info)
        self._add_history_entry("preprocessing", step_info)

    def set_business_context(self, context):
        """Store business context information."""
        st.session_state.business_context = context
        self._add_history_entry(
            "business_context",
            {
                "summary": "Business context defined",
                "domain": context.get("domain", "Unknown"),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def add_analysis(self, analysis):
        """Add an analysis to history."""
        st.session_state.analyses.append(analysis)
        self._add_history_entry(
            "analysis",
            {
                "type": analysis.get("type"),
                "summary": analysis.get("summary", "Analysis completed"),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def add_visualization(self, viz):
        """Add a visualization to collection."""
        # Generate a unique ID if not provided
        if "id" not in viz:
            viz["id"] = f"viz_{len(st.session_state.visualizations)}"

        st.session_state.visualizations.append(viz)
        self._add_history_entry(
            "visualization",
            {
                "id": viz["id"],
                "type": viz.get("type"),
                "title": viz.get("title", "Untitled"),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def set_dashboard_config(self, config):
        """Store dashboard configuration."""
        st.session_state.dashboard_config = config
        self._add_history_entry(
            "dashboard",
            {
                "summary": "Dashboard configured",
                "layout_sections": len(config.get("layout", [])),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _add_history_entry(self, step_type, data):
        """Add an entry to the history log."""
        entry = {
            "step": step_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state.history.append(entry)

    def navigate_to(self, step):
        """Navigate to a specific step in the workflow."""
        st.session_state.step = step

    def export_configuration(self):
        """Export the full dashboard configuration."""
        config = {
            "metadata": {
                "project_name": st.session_state.dataset_name,
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
            },
            "data_source": {"type": "csv", "filename": st.session_state.dataset_name},
            "preprocessing": st.session_state.preprocessing,
            "business_context": st.session_state.business_context,
            "analyses": st.session_state.analyses,
            "visualizations": st.session_state.visualizations,
            "dashboard": st.session_state.dashboard_config,
        }

        return json.dumps(config, indent=2)
