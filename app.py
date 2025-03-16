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

            if st.session_state.profile is None:
                disabled_options.extend(nav_options[1:])
            elif not st.session_state.preprocessing:
                disabled_options.extend(nav_options[2:])
            elif not st.session_state.business_context:
                disabled_options.extend(nav_options[3:])
            elif not st.session_state.analyses:
                disabled_options.extend(nav_options[4:])
            elif not st.session_state.visualizations:
                disabled_options.extend(nav_options[5:])
            elif not st.session_state.dashboard_config:
                disabled_options.extend(nav_options[6:])

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
        st.header("Dashboard Assembly")
        st.info("This feature will be implemented next.")
        # Provide a way to continue to the next step for now
        if st.button("Continue to Export"):
            st.session_state.step = "export"
            st.rerun()
    elif st.session_state.step == "export":
        st.header("Export")
        st.info("This feature will be implemented next.")


if __name__ == "__main__":
    main()
