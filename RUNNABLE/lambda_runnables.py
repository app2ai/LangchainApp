from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

p1 = PromptTemplate(
    template='Write an essay on {topic}',
    input_variables=['topic']
)

strParser = StrOutputParser()

def numberOfWordsInEssay(text):
    return len(text.split())

gen_essay = RunnableSequence(p1, model, strParser)

rnbl = RunnableParallel({
    'essay': RunnablePassthrough(),
    'word_count': RunnableLambda(numberOfWordsInEssay)
})

final_run = RunnableSequence(gen_essay, rnbl)

result = final_run.invoke({'topic':'Sanatan Dharm'})

print(result)

# OUTPUT
#{'essay': 'Sanatan Dharm, also known as Hinduism, is one of the oldest and most complex religions in the world. It is a way of life
#  that has been practiced for over 4,000 years and is still widely practiced today. At its core, Sanatan Dharm is a system of
#  moral and spiritual development that seeks to unite the individual with the universe and achieve spiritual liberation.\n\nThe
#  concept of Sanatan Dharm is rooted in the idea that the ultimate reality, or Brahman, is beyond human comprehension.
#  It is a complex, multifaceted, and ever-evolving concept that encompasses the entire universe and all of its manifestations. 
# The goal of Sanatan Dharm is to achieve a state of unity with this ultimate reality, which is often referred to as Moksha
#  or Nirvana.\n\nAt the heart of Sanatan Dharm is the concept of the Chaturvarna, or the four stages of life. These stages
#  are:\n\n1. Brahmacharya, or the stage of student life, where one studies and learns from the world.\n2. Grihastha, or the 
# stage of married life, where one settles down and raises a family.\n3. Vanaprastha, or the stage of retirement, where one
#  withdraws from the world and focuses on spiritual development.\n4. Sanyasa, or the stage of renunciation, where one renounces 
# worldly attachments and focuses solely on spiritual development.\n\nSanatan Dharm also emphasizes the importance of the three Gunas, or qualities, which are:\n\n1. Sattva, or the quality of purity and goodness.\n2. Rajas, or
#  the quality of passion and activity.\n3. Tamas, or the quality of darkness and ignorance.\n\nThe goal of Sanatan Dharm is to balance these three Gunas and achieve 
# a state of spiritual equilibrium.\n\nSanatan Dharm is also deeply rooted in the concept of Karma, or the law of cause and effect. According to this philosophy, every action has consequences, and one must work to
#  purify their karma in order to achieve spiritual liberation.\n\nIn addition to these philosophical concepts, Sanatan Dharm is also a rich and diverse cultural tradition that encompasses a wide 
# range of practices and rituals. These include the worship of various deities, the performance of daily pujas, or worship rituals, and the observance of various festivals
#  and celebrations.\n\nDespite its complexity and diversity, Sanatan Dharm is often misunderstood and misrepresented by outsiders. It is often seen as a simplistic or primitive religion, and
#  its many philosophical and',
#  'word_count': 392}
