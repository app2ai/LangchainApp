from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch
from pydantic import Field, BaseModel
from typing import Literal

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
#     task = "text-generation"
# )

# model = ChatHuggingFace(llm = llm)
model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

class BotDecision(BaseModel):
    intent: Literal['Booking','Cancel','General'] = Field(description='When user ask query, bot classify that query into these')

user_query = 'I want to book a Hotel in North Goa.'

pyParser = PydanticOutputParser(pydantic_object = BotDecision)
strParser = StrOutputParser()

queryTemplate = PromptTemplate(
    template='User query: {user_query}.\n {format_instruction}',
    input_variables=['user_query'],
    partial_variables={'format_instruction': pyParser.get_format_instructions()}
)

find_intent = RunnableSequence(queryTemplate, model, pyParser)

print(find_intent.invoke({'user_query':user_query}))

bookingTemplate = PromptTemplate(
    template='Guide user to book there requirement as given in \n{user_query}',
    input_variables=['user_query']
)

cancelTemplate = PromptTemplate(
    template='Guide user to \n{user_query}',
    input_variables=['user_query']
)

generalQTemplate = PromptTemplate(
    template='Write an appropriate response to user on \n{user_query}',
    input_variables=['user_query']
)

cRunnable = RunnableBranch(
    (lambda x: x.intent == 'Booking', RunnableSequence(bookingTemplate, model, strParser)),
    (lambda x: x.intent == 'Cancel', RunnableSequence(cancelTemplate, model, strParser)),
    (RunnableSequence(generalQTemplate, model, strParser))
)

fRun = RunnableSequence(find_intent, cRunnable)

ff = fRun.invoke({'user_query': user_query})

print(ff)
