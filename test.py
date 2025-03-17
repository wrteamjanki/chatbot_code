import toml

# Load API Key from secrets.toml
def load_api_key():
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        api_key = secrets["general"].get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found. Please check your secrets.toml file.")
        print(f"✅ GEMINI_API_KEY loaded successfully: {api_key}")
    except FileNotFoundError:
        print("❌ secrets.toml file not found. Ensure it's located in the .streamlit directory.")
    except KeyError:
        print("❌ 'general' section not found in secrets.toml.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    load_api_key()

