from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model_name="claude-sonnet-4-5", temperature=0.5, max_tokens_to_sample=15)

result = model.invoke("Who is better America or Israil?")

print(result.content)