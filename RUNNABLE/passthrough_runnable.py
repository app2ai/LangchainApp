from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

p1 = PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

p2 = PromptTemplate(
    template='Give an explanation for a {joke}',
    input_variables=['joke']
)

strParser = StrOutputParser()

generate_joke = RunnableSequence(p1, model, strParser)

explanation = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation' : RunnableSequence(p2, model, strParser)
})

result = RunnableSequence(generate_joke, explanation)

fResult = result.invoke({'topic':'War'})

print(fResult)
