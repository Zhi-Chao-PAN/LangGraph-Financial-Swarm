import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
print("DEBUG: OPENAI_API_KEY starts with:", os.getenv("OPENAI_API_KEY")[:10] if os.getenv("OPENAI_API_KEY") else "None")

try:
    llm = ChatOpenAI(model="gpt-4o")
    res = llm.invoke("Hi")
    print("Success:", res.content)
except Exception as e:
    print("Connection Error Type:", type(e))
    print("Connection Error Message:", str(e))
