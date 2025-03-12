import google.generativeai as genai
import os

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDRrRi3qHv3gcFA-IAtfsjICXzHCb9HX5E"  # Replace with your actual API key

# Configure the API key for generative AI
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# List all available models
models = genai.list_models()

# Print the models to verify
for model in models:
    print(model)
