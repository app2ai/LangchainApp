from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="What are core component of {language} programming?",
    input_variables=['language']
)

prompt2 = PromptTemplate(
    template="Write a 5 line summary on, \n{result}",
    input_variables=['result']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'language':'Kotlin'})

print(result)

# print(chain.get_graph().draw_mermaid())

# OUTPUT -> 
# Here is a 5-line summary:
# Kotlin is a modern, statically-typed programming language that runs on the Java Virtual Machine (JVM). 
# It features a concise and safe syntax, with a strong type system that prevents null pointer exceptions.
# Kotlin also supports extension functions, higher-order functions, and lambda expressions, making it a versatile language.
# Additionally, it has built-in null safety features and interoperability with Java, making it easy to use with existing code.
# Overall, Kotlin provides a modern and expressive way to write software with a rich standard library.
