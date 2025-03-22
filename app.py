import streamlit as st
from modules.session import SessionState
from modules.data_ingestion import (
    render_upload_interface,
    run_data_profiling,
    display_data_profile,
)
from modules.preprocessing import render_preprocessing_interface
from modules.business_understanding import render_business_understanding
from modules.analysis import render_analysis_interface
from modules.visualization import render_visualization_interface
from modules.dashboard_assembly import render_dashboard_assembly
from modules.export import render_export_interface
from utils.data_utils import apply_preprocessing


def handle_upload_step():
    """Handle the data upload step."""
    # Display file upload interface
    upload_result = render_upload_interface()

    # If data is uploaded, process it
    if upload_result is not None:
        df, filename = upload_result

        # Save to session state if not already there
        if (
            st.session_state.dataset is None
            or st.session_state.dataset_name != filename
        ):
            # Store in session
            session = SessionState()
            session.set_dataset(df, filename)

            # Run data profiling
            st.subheader("Data Profiling")
            st.info(
                "Analyzing your dataset to understand its structure and potential..."
            )

            profile = run_data_profiling(df)
            session.set_profile(profile)

            # Force rerun to update UI
            st.rerun()
        else:
            # Show data profile if already analyzed
            if st.session_state.profile:
                display_data_profile(st.session_state.profile)

                # Next step button
                if st.button("Continue to Data Preprocessing"):
                    st.session_state.step = "data_preprocessing"
                    st.rerun()


def handle_preprocessing_step():
    """Handle the data preprocessing step."""
    # Ensure we have data and profile
    if st.session_state.dataset is None or st.session_state.profile is None:
        st.error("Please upload and profile a dataset first.")
        if st.button("Go to Data Upload"):
            st.session_state.step = "upload"
            st.rerun()
        return

    # Render preprocessing interface
    preprocessing_steps = render_preprocessing_interface(
        profile=st.session_state.profile, df=st.session_state.dataset
    )

    if preprocessing_steps is not None:
        # If preprocessing steps were selected
        session = SessionState()

        if preprocessing_steps:
            # Apply preprocessing
            with st.spinner("Applying preprocessing steps..."):
                processed_df = apply_preprocessing(
                    st.session_state.dataset, preprocessing_steps
                )

                # Update dataset with processed version
                session.set_dataset(processed_df, st.session_state.dataset_name)

                # Store preprocessing steps
                for step in preprocessing_steps:
                    session.add_preprocessing_step(step)

                # Refresh profile to capture new column types after preprocessing
                updated_profile = run_data_profiling(processed_df)
                session.set_profile(updated_profile)

                st.success("Preprocessing applied successfully!")

        # Move to next step
        st.session_state.step = "business_understanding"
        st.rerun()


def handle_business_understanding_step():
    """Handle the business understanding step."""
    # Ensure we have data
    if st.session_state.dataset is None:
        st.error("Please upload a dataset first.")
        if st.button("Go to Data Upload"):
            st.session_state.step = "upload"
            st.rerun()
        return

    # Render business understanding interface
    business_context = render_business_understanding(
        df=st.session_state.dataset, profile=st.session_state.profile
    )

    if business_context is not None:
        # Store business context
        session = SessionState()
        session.set_business_context(business_context)

        # Move to next step
        st.session_state.step = "advanced_analysis"
        st.rerun()


def handle_analysis_step():
    """Handle the advanced analysis step."""
    # Ensure we have data and business context
    if st.session_state.dataset is None or not st.session_state.business_context:
        st.error("Please complete the previous steps first.")
        if st.button("Go to Business Understanding"):
            st.session_state.step = "business_understanding"
            st.rerun()
        return

    # Render analysis interface
    analysis_results = render_analysis_interface(
        df=st.session_state.dataset, business_context=st.session_state.business_context
    )

    if analysis_results is not None:
        # Store complete analysis results
        st.session_state.analysis_results = analysis_results

        # Store analyses summaries
        session = SessionState()
        for analysis in analysis_results:
            session.add_analysis(
                {
                    "type": analysis.get("type"),
                    "title": analysis.get("title"),
                    "summary": analysis.get("summary"),
                }
            )

        # Move to next step
        st.session_state.step = "visualization_generation"
        st.rerun()


def handle_visualization_step():
    """Handle the visualization generation step."""
    # Ensure we have data and analyses
    if st.session_state.dataset is None or not st.session_state.analyses:
        st.error("Please complete the previous steps first.")
        if st.button("Go to Advanced Analysis"):
            st.session_state.step = "advanced_analysis"
            st.rerun()
        return

    # Render visualization interface
    visualizations = render_visualization_interface(
        df=st.session_state.dataset,
        analysis_results=st.session_state.analysis_results,
        business_context=st.session_state.business_context,
    )

    if visualizations is not None:
        # Store visualizations
        session = SessionState()
        for viz in visualizations:
            session.add_visualization(viz)

        # Move to next step
        st.session_state.step = "dashboard_assembly"
        st.rerun()


def handle_dashboard_assembly_step():
    """Handle the dashboard assembly step."""
    # Ensure we have data and visualizations
    if st.session_state.dataset is None or not st.session_state.visualizations:
        st.error("Please complete the previous steps first.")
        if st.button("Go to Visualization Generation"):
            st.session_state.step = "visualization_generation"
            st.rerun()
        return

    # Render dashboard assembly interface
    dashboard_config = render_dashboard_assembly(
        df=st.session_state.dataset,
        visualizations=st.session_state.visualizations,
        business_context=st.session_state.business_context,
    )

    if dashboard_config is not None:
        # Store dashboard configuration
        session = SessionState()
        session.set_dashboard_config(dashboard_config)

        # Move to next step
        st.session_state.step = "export"
        st.rerun()


def handle_export_step():
    """Handle the export step."""
    # Ensure we have dashboard configuration
    if st.session_state.dataset is None or not st.session_state.dashboard_config:
        st.error("Please complete the Dashboard Assembly step first.")
        if st.button("Go to Dashboard Assembly"):
            st.session_state.step = "dashboard_assembly"
            st.rerun()
        return

    # Render export interface
    export_completed = render_export_interface(
        dashboard_config=st.session_state.dashboard_config,
        visualizations=st.session_state.visualizations,
        business_context=st.session_state.business_context,
        analyses=st.session_state.analyses,
    )

    if export_completed:
        # Show completion message
        st.success(
            "🎉 Congratulations! You've successfully completed the dashboard generation process."
        )

        # Provide option to start over
        if st.button("Start a New Dashboard"):
            # Reset session state
            for key in list(st.session_state.keys()):
                if key != "initialized":
                    del st.session_state[key]

            # Initialize new session
            session = SessionState()

            # Go back to upload step
            st.session_state.step = "upload"
            st.rerun()


def main():
    """Main application entry point."""
    # Initialize session state
    session = SessionState()

    # Navigation sidebar
    with st.sidebar:
        st.title("📊 Dashboard Generator")
        st.markdown("---")

        # Generate navigation based on available steps
        if st.session_state.dataset is not None:
            nav_options = [
                "Data Upload",
                "Data Preprocessing",
                "Business Understanding",
                "Advanced Analysis",
                "Visualization Generation",
                "Dashboard Assembly",
                "Export",
            ]

            # Enable navigation based on progress
            disabled_options = []
            current_step = st.session_state.step
            current_step_index = -1

            # Find current step index
            for i, option in enumerate(nav_options):
                if option.lower().replace(" ", "_") == current_step:
                    current_step_index = i
                    break

            # Disable steps that come after unreached milestones
            if st.session_state.profile is None:
                disabled_options.extend(nav_options[1:])
            elif not st.session_state.preprocessing and current_step_index < 1:
                disabled_options.extend(nav_options[2:])
            elif not st.session_state.business_context and current_step_index < 2:
                disabled_options.extend(nav_options[3:])
            elif not st.session_state.analyses and current_step_index < 3:
                disabled_options.extend(nav_options[4:])
            elif not st.session_state.visualizations and current_step_index < 4:
                disabled_options.extend(nav_options[5:])
            elif not st.session_state.dashboard_config and current_step_index < 5:
                disabled_options.extend(nav_options[6:])

            # Always enable current and previous steps
            for i in range(current_step_index + 1):
                if nav_options[i] in disabled_options:
                    disabled_options.remove(nav_options[i])

            # Show progress
            st.progress((nav_options.index(nav_options[0]) + 1) / len(nav_options))

            # Navigation buttons
            for option in nav_options:
                disabled = option in disabled_options
                if st.button(option, disabled=disabled, key=f"nav_{option}"):
                    st.session_state.step = option.lower().replace(" ", "_")
                    st.rerun()

        st.markdown("---")
        st.markdown("Powered by AI")

    # Main content area based on current step
    if st.session_state.step == "upload":
        handle_upload_step()
    elif st.session_state.step == "data_preprocessing":
        handle_preprocessing_step()
    elif st.session_state.step == "business_understanding":
        handle_business_understanding_step()
    elif st.session_state.step == "advanced_analysis":
        handle_analysis_step()
    elif st.session_state.step == "visualization_generation":
        handle_visualization_step()
    elif st.session_state.step == "dashboard_assembly":
        handle_dashboard_assembly_step()
    elif st.session_state.step == "export":
        handle_export_step()


if __name__ == "__main__":
    main()
