import json
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

# Initialize OpenAI client
# Using only the API key parameter to avoid compatibility issues
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("WARNING: OPENAI_API_KEY not found. Some functionality will be limited.")


def get_chat_completion(messages, temperature=0.0, response_format=None):
    """
    Get a completion from the OpenAI chat model.

    Args:
        messages (list): List of message dictionaries in OpenAI format
        temperature (float): Control randomness (0.0 to 1.0)
        response_format (dict, optional): Format for response (e.g., {"type": "json_object"})

    Returns:
        str: Model response content
    """
    if client is None:
        print("WARNING: OpenAI client not initialized. Cannot get completion.")
        # Return mock response for testing
        return '{"dataset_purpose": "This is mock data for testing", "column_descriptions": {}, "potential_analyses": [], "preprocessing_suggestions": []}'

    try:
        kwargs = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error getting completion: {e}")
        return '{"error": "Failed to get completion from OpenAI"}'


def analyze_data_schema(df_sample, df_info):
    """
    Use OpenAI to analyze DataFrame schema and provide understanding.

    Args:
        df_sample (str): Stringified DataFrame sample (head/tail)
        df_info (dict): Information about the DataFrame (shape, dtypes, etc.)

    Returns:
        dict: Analysis results including column descriptions and recommendations
    """
    if client is None:
        print("WARNING: OpenAI client not initialized. Using basic analysis.")
        # Return basic analysis without OpenAI
        columns = df_info.get("columns", [])
        column_descriptions = {
            col: f"Column containing data of type {df_info.get('dtypes', {}).get(col, 'unknown')}"
            for col in columns
        }

        return {
            "dataset_purpose": "Dataset purpose could not be determined without OpenAI analysis",
            "column_descriptions": column_descriptions,
            "potential_analyses": [
                "Basic statistical analysis",
                "Data visualization",
                "Correlation analysis for numeric columns",
            ],
            "preprocessing_suggestions": [
                {
                    "type": "missing_values",
                    "columns": [
                        col
                        for col, count in df_info.get("missing_values", {}).items()
                        if count > 0
                    ],
                }
            ],
            "visualization_ideas": [
                "Bar charts for categorical data",
                "Line charts for time series data",
                "Scatter plots for relationships between numeric columns",
            ],
        }

    messages = [
        {
            "role": "system",
            "content": "You are a data analyst AI that helps understand datasets.",
        },
        {
            "role": "user",
            "content": f"""
        Analyze this dataset sample and provide information about its structure, purpose, and potential insights.
        
        DataFrame Info:
        {json.dumps(df_info, indent=2)}
        
        DataFrame Sample:
        {df_sample}
        
        Provide your analysis in JSON format with these sections:
        1. dataset_purpose: Your best guess of what this dataset represents
        2. column_descriptions: For each column, describe its meaning and data type
        3. potential_analyses: List of analytical approaches appropriate for this data
        4. preprocessing_suggestions: What preprocessing steps might be needed
        5. visualization_ideas: Visualization types that would be effective
        """,
        },
    ]

    try:
        result = get_chat_completion(
            messages, temperature=0.2, response_format={"type": "json_object"}
        )
        return json.loads(result)
    except json.JSONDecodeError:
        # Fallback for non-JSON responses
        return {"error": "Failed to parse JSON response"}
