from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Detect provider based on available key
if GEMINI_API_KEY:
    PROVIDER = "gemini"
elif OPENAI_API_KEY:
    PROVIDER = "openai"
else:
    PROVIDER = None
