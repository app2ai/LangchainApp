from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-3.5-turbo')

template = PromptTemplate(
    template = "Tell me about {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

# This is a simple chain
chain = template | model | parser

result = chain.invoke({'topic':'Agents in AI in Short 3 lines'})

print(result)

# Chains graph
print(chain.get_graph().draw_mermaid())

# O/P ---------------------------------------
# PromptInput([PromptInput]):::first
# PromptTemplate(PromptTemplate)
# ChatOpenAI(ChatOpenAI)
# StrOutputParser(StrOutputParser)
# StrOutputParserOutput([StrOutputParserOutput]):::last
# PromptInput --> PromptTemplate;
# PromptTemplate --> ChatOpenAI;
# StrOutputParser --> StrOutputParserOutput;
# ChatOpenAI --> StrOutputParser;
