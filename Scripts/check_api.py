import os
import requests
from dotenv import load_dotenv

load_dotenv("Config/api_keys.env")

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
)

print("Status code:", response.status_code)
print("\nRate limit headers:")
for header, value in response.headers.items():
    if "ratelimit" in header.lower():
        print(f"  {header}: {value}")

if response.status_code != 200:
    print("\nError body:")
    print(response.json())