# src/generation.py
import os
import requests
import openai

# --- Gemini API wrapper ---
def generate_with_gemini(prompt, api_key, max_tokens=500):
    """
    Sends a prompt to Gemini API and returns the generated answer as a string.
    Replace 'GEMINI_ENDPOINT' with actual Gemini API endpoint.
    """
    GEMINI_ENDPOINT = "https://api.gemini.com/v1/generate" 

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("text", "[Gemini returned no text]")
    except Exception as e:
        return f"[Gemini API error]: {str(e)}"


# --- OpenAI API wrapper ---
def generate_with_openai(prompt, api_key, model="gpt-4", max_tokens=500):
    """
    Sends a prompt to OpenAI API and returns the generated answer as a string.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            api_key=api_key
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenAI API error]: {str(e)}"


# --- Unified generation function ---
from config import PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY

def generate_answer(prompt):
    """
    Returns the generated answer using the selected provider in config.py
    """
    if PROVIDER.lower() == "gemini":
        return generate_with_gemini(prompt, GEMINI_API_KEY)
    elif PROVIDER.lower() == "openai":
        return generate_with_openai(prompt, OPENAI_API_KEY)
    else:
        raise ValueError("No valid provider set in config.py")
