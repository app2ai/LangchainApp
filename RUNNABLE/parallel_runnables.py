from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

# model = ChatOpenAI(model='gpt-5-nano-2025-08-07')
llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

tempTweet = PromptTemplate(
    template='Generate a {length} tweet related to {topic} for X',
    input_variables=['length', 'topic']
)

tempLinkedIn = PromptTemplate(
    template='Generate a {length} post related to {topic} for LinkedIn',
    input_variables=['length', 'topic']
)

strParser = StrOutputParser()

# OP -> Dictionary
runn = RunnableParallel({
    'tweet': RunnableSequence(tempTweet, model, strParser),
    'post': RunnableSequence(tempLinkedIn, model, strParser)
    })

res = runn.invoke({'length':'short', 'topic':'AI Agents'})

print(res)
