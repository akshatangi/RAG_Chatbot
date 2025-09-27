from src.config import PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY

def generate_answer(prompt: str):
    if PROVIDER == "gemini":
        return generate_with_gemini(prompt, GEMINI_API_KEY)
    elif PROVIDER == "openai":
        return generate_with_openai(prompt, OPENAI_API_KEY)
    else:
        raise ValueError("No valid provider set in config.")
