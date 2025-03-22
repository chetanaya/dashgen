# AI-Driven Dashboard Generator

A powerful application that automatically analyzes your data and generates interactive dashboards with minimal user input.

## Overview

The AI-Driven Dashboard Generator is a tool that uses AI to understand your data, identify key insights, and create visualizations automatically. The application guides you through a step-by-step process, from data upload to dashboard export, with AI assisting at every stage.

## Features

- **Intelligent Data Profiling**: Automatically detects data types, relationships, and quality issues
- **Smart Preprocessing**: Recommends and applies appropriate data cleaning and transformation steps
- **Business Context Detection**: Identifies the business domain and key metrics from your data
- **Advanced Analysis**: Performs statistical analyses and generates insights
- **Visualization Recommendations**: Creates appropriate visualizations based on data characteristics
- **Interactive Dashboard**: Assembles visualizations into a cohesive dashboard with filters
- **Exportable Package**: Generates a complete Streamlit application that can be deployed independently
- **Easy Updates**: Exported dashboards support data updates without code modifications

## Project Structure

```
dashboard_generator/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── agents/                 # AI agent implementations
│   ├── data_profiler.py    # Data understanding agent
│   └── preprocessing.py    # Preprocessing recommendation agent
├── modules/                # Application modules
│   ├── data_ingestion.py   # CSV upload and profiling
│   ├── preprocessing.py    # Data preprocessing pipeline
│   ├── business_understanding.py # Business context identification
│   ├── analysis.py         # Advanced analysis module
│   ├── visualization.py    # Visualization generation
│   ├── dashboard_assembly.py # Dashboard layout and configuration
│   ├── export.py           # Dashboard export functionality
│   └── session.py          # Session state management
├── utils/                  # Utility functions
│   ├── data_utils.py       # Data handling utilities
│   └── openai_utils.py     # OpenAI API integration
└── config/                 # Configuration settings
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chetanaya/dashgen.git
   cd dashgen
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Follow the step-by-step process:
   - **Upload Data**: Upload your CSV file
   - **Data Preprocessing**: Apply recommended transformations to clean and prepare your data
   - **Business Understanding**: Define the business context of your data
   - **Advanced Analysis**: Run analyses to discover insights
   - **Visualization Generation**: Create and select visualizations
   - **Dashboard Assembly**: Organize visualizations into a dashboard layout
   - **Export**: Download your dashboard as a standalone application

## Exported Dashboards

The exported dashboard package includes:

- **app.py**: Streamlit application with preprocessing logic
- **Original data file**: Your uploaded CSV
- **dashboard_config.json**: Dashboard configuration
- **README.md**: Instructions for running and updating the dashboard
- **requirements.txt**: Required Python packages
- **preprocessing_documentation.md**: Details of applied preprocessing steps

To update the dashboard with new data:

1. Replace the CSV file with your updated data
2. Run the dashboard with `streamlit run app.py`
3. The preprocessing steps will be automatically applied to the new data

## Dependencies

- streamlit==1.31.1
- pandas==2.2.0
- numpy==1.26.3
- plotly==5.18.0
- altair==5.2.0
- openai==1.12.0
- python-dotenv==1.0.1
- scipy==1.12.0
- scikit-learn==1.4.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Powered by OpenAI's large language models
- Built with Streamlit for interactive web applications
- Uses Plotly for dynamic data visualizations