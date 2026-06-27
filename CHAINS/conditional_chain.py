from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano-2025-08-07')

feedback = "Macbook neo is comeup with great feature with less price."

class Feedback(BaseModel):
    userSentiment: Literal['positive', 'negative'] = Field(description='Classify user feedback into positive or negative sentiment')

pydanticParser = PydanticOutputParser(pydantic_object=Feedback)

feedbackTemplate = PromptTemplate(
    template='User feedback: {feedback}.\n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydanticParser.get_format_instructions()}
)

sentimentChain = feedbackTemplate | model | pydanticParser

print(sentimentChain.invoke({'feedback': feedback}))

strParser = StrOutputParser()

posTemplate = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n{feedback}',
    input_variables=['feedback']
)

negTemplate = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n{feedback}',
    input_variables=['feedback']
)

nChain = negTemplate | model | strParser
pChain = posTemplate | model | strParser

conditionalChain = RunnableBranch(
    (lambda x:x.userSentiment=='positive', pChain),
    (lambda x:x.userSentiment=='negative', nChain),
    RunnableLambda(lambda x: 'Could not find right sentiment')
)

chain = sentimentChain | conditionalChain

res = chain.invoke({'feedback': feedback})

print(res)

# OUTPUT:
# userSentiment='positive'
# Thank you so much for sharing your positive feedback! We’re thrilled to hear you’re happy with our service.
# We truly appreciate your support and will continue to strive for excellence. If there’s anything else we can 
# do for you or any suggestions you have, please don’t hesitate to letus know.