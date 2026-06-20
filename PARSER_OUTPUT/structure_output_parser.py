from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

sParser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(
        name="top_all_rounder_cricketer",
        description="Top 5 all-rounder cricketers with name, country and ranking"
    )
])


templatePrompt = PromptTemplate(
    template="List down icc ranking of top 5 {cricker_type} contains Name, Country and Ranking number. \n{format_instructions}",
    input_variables=['cricker_type'],
    partial_variables={"format_instructions": sParser.get_format_instructions()}   
)

res = model.invoke(templatePrompt.format(cricker_type="All-rounder"))

finalRes = sParser.parse(res.content)

print(finalRes)

# Output:
# {'top_all_rounder_cricketer': [
#   {'name': 'Ben Stokes', 'country': 'England', 'ranking': 1}, 
#   {'name': 'Jos Buttler', 'country': 'England', 'ranking': 2}, 
#   {'name': 'Hardik Pandya', 'country': 'India', 'ranking': 3}, 
#   {'name': 'Andre Russell', 'country': 'West Indies', 'ranking': 4}, 
#   {'name': 'Dwayne Bravo', 'country': 'West Indies', 'ranking': 5}]
# }
