from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person contains first name and last name in single string")
    age: int = Field(gt = 18, description="Age of the person")
    city: str = Field(description="City of the person")

pParser = PydanticOutputParser(pydantic_object=Person) 

template = PromptTemplate(
    template="Generate name of person belongs to {country}.\n {format_instructions}",
    input_variables=['country'],
    partial_variables={"format_instructions": pParser.get_format_instructions()}
)

print(template)
# Actual Prompt Input
# input_variables=['country'] input_types={} partial_variables={'format_instructions': 'The output should be formatted as a
#  JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema 
# {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, 
# "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object 
# {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": 
# {"name": {"description": "Name of the person contains first name and last name", "title": "Name", "type": "string"}, 
# "age": {"description": "Age of the person", "exclusiveMinimum": 18, "title": "Age", "type": "integer"}, 
# "city": {"description": "City of the person", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}\n```'} 
# template='Generate name of person belongs to {country}.\n {format_instructions}'

chain = template | model | pParser

res = chain.invoke({'country': 'India'})

print(res)
# OUTPUT -> name='RohanDesai' age=25 city='Mumbai'

print(res.json()) 
# OUTPUT -> {"name":"Rohan Desai","age":25,"city":"Ahmedabad"}
