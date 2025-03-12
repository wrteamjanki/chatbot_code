import os
import json
import google.generativeai as genai

def setup_gemini_api():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
            gemini_api_key = config_data.get("GEMINI_API_KEY")
            if gemini_api_key:
                os.environ["GEMINI_API_KEY"] = gemini_api_key
                genai.configure(api_key=gemini_api_key)
            else:
                raise ValueError("GEMINI_API_KEY is not set in config.json")
    else:
        raise FileNotFoundError("config.json file not found.")

setup_gemini_api()
