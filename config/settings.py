import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"

# Data ingestion settings
MAX_FILE_SIZE_MB = 10
SAMPLE_SIZE = 5000  # For large files

# Session settings
SESSION_EXPIRY_DAYS = 7

# Visualization settings
DEFAULT_COLOR_SCHEME = "blues"
MAX_VISUALIZATIONS = 10
