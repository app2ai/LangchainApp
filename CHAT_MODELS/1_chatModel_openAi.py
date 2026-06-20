from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-2025-04-14")

result = model.invoke("Who is richest person in the world?")

print(result)