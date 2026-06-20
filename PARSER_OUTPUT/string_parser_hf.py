from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv, parser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Deepseek LLM API endpoint
llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

# prompt 1 for asking the topic of the text
promptTopic = PromptTemplate(
    template="Give me detailed knowledge about {topic}",
    input_variables=["topic"]
)

# prompt 2 for asking the topic of the text
promptSummary = PromptTemplate(
    template="Summarize {text} in 4 lines",
    input_variables=["text"]
)

stParser = StrOutputParser()

chain = promptTopic | model | stParser | promptSummary | model | stParser

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)

# Output: "Artificial Intelligence (AI) is a branch of computer science that focuses on creating intelligent machines capable 
# of performing tasks that typically require human intelligence. It encompasses various subfields such as machine learning,
#  natural language processing, computer vision, and robotics. AI systems can analyze large amounts of data, recognize patterns,
#  and make decisions based on that information. The goal of AI is to develop systems that can learn from experience, adapt to
#  new inputs, and perform tasks autonomously. AI has applications in various industries including healthcare, finance, 
# transportation, and entertainment."