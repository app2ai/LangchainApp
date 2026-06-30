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

# OUTPUT
# {'joke': 'Why did the war go to therapy?\n\nBecause it was feeling "battled" and wanted to "disarm" its emotions!',
#  'explanation': 'I see what you did there!\n\nThe pun is that the war, as in a conflict or a battle, went to therapy 
#  because it was "feeling battled" (i.e., feeling overwhelmed and exhausted) and wanted to "disarm" (i.e., let go of) 
#  its emotions.\n\nIn other words, the war is using a play on words to describe its emotional state, using military terminology
#  to express its feelings. It\'s a clever and humorous way to poke fun at the idea of a conflict seeking help for its emotional 
# well-being.\n\nWell done, punster!'
# }