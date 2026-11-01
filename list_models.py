import os
from dotenv import load_dotenv

load_dotenv()

# Prevent the old key from being picked up
os.environ.pop("GOOGLE_API_KEY", None)

from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Available Gemini models:\n")

for model in client.models.list():
    print(model.name)