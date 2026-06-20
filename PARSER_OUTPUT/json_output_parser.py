from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

chatModel = ChatHuggingFace(llm = llm)

jsonParser = JsonOutputParser()

# prompt for asking the topic of the text
promptTopic = PromptTemplate(
    template= "List down icc ranking of top 5 Cricket batsman contains Name, Country and Ranking number. \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": jsonParser.get_format_instructions()}
)

# NOTE: WE can not provide explicit schema to json parser. this is limitation of JsonOutputParser.
chain = promptTopic | chatModel | jsonParser

result = chain.invoke({})

#final_result = jsonParser.parse(result)

print(result)

# Output: JSON Format
# {'icc_ranking': [{'name': 'Kohli, Virat M', 'country': 'India', 'ranking_number': 1},
# {'name': 'Root, J E', 'country': 'England', 'ranking_number': 2}, 
# {'name': 'Kohli, R', 'country': 'India', 'ranking_number': 3}, 
# {'name': 'Labuschagne, M', 'country': 'Australia', 'ranking_number': 4}, 
# {'name': 'Babar, Azam, M', 'country': 'Pakistan', 'ranking_number': 5}]
# }
