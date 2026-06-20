from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

model = OpenAI(model='gpt-3.5-turbo-instruct')

result = model.invoke('What is Gen AI explain in 40 words?') 

print(result)