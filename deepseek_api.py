import os
import toml
import requests

# Load API Key from secrets.toml (adjust path if needed)
secrets = toml.load(".streamlit/secrets.toml")
api_key = secrets["general"].get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Please check your secrets.toml file.")

# Define the API endpoint and headers
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def query_deepseek(user_question):
    # Enhanced prompt combining system and user messages
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a knowledgeable assistant with access to detailed information from a collection of documents. "
                "Provide concise and direct answers. If you don't have live data (e.g., current weather), say 'I don't have that information.'"
            )
        },
        {"role": "user", "content": user_question}
    ]

    data = {
        "model": "deepseek/deepseek-r1-zero:free",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        print("Full API Response:", result)  # Debug print
        message = result["choices"][0]["message"]
        # Try to get the direct answer; if empty, fall back to reasoning
        answer = message.get("content", "").strip()
        if not answer:
            answer = message.get("reasoning", "").strip()
        return answer
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    question = input("Ask Deepseek: ")
    answer = query_deepseek(question)
    if answer:
        print("Deepseek Response:", answer)
    else:
        print("No response received.")
