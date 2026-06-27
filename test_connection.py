from openai import OpenAI
from dotenv import load_dotenv
import os
##this file was made to test the connection to the DeepSeek API.
load_dotenv('Config/api_keys.env')

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": 'Connection successful!'}
        ],
        temperature=0.3
    )
    
    print("DeepSeek API Connection Successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f" Connection Failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check your API key in config/api_keys.env")
    print("2. Verify you have internet connection")
    print("3. Make sure openai package is installed: pip install openai")
