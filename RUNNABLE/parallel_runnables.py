from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

# model = ChatOpenAI(model='gpt-5-nano-2025-08-07')
llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

tempTweet = PromptTemplate(
    template='Generate a {length} tweet related to {topic} for X',
    input_variables=['length', 'topic']
)

tempLinkedIn = PromptTemplate(
    template='Generate a {length} post related to {topic} for LinkedIn',
    input_variables=['length', 'topic']
)

strParser = StrOutputParser()

# OP -> Dictionary
runn = RunnableParallel({
    'tweet': RunnableSequence(tempTweet, model, strParser),
    'post': RunnableSequence(tempLinkedIn, model, strParser)
    })

res = runn.invoke({'length':'short', 'topic':'AI Agents'})

print(res)

# OUTPUT
# {
# 'tweet': 'Here\'s a short tweet about AI Agents for X:\n\n"Intelligent AI Agents are revolutionizing the field of X! With their ability to learn,
#   reason, and adapt, they\'re transforming the way we [X] and opening up new possibilities for innovation. #AIX #X #AI #MachineLearning"',
# 'post': 'Here\'s a potential short post related to AI Agents for LinkedIn:\n\n**Title:** "Revolutionizing Sales with AI Agents: 
#   The Future of B2B Lead Generation"\n\n**Post:**\n\n"Are you tired of cold calling and emailing? Do you struggle to identify the most promising leads for your business?
#   AI is here to change the game!\n\nIntroducing AI Agents: the next generation of sales intelligence. These intelligent agents
#   use machine learning algorithms to analyze large datasets, identify patterns, and predict customer behavior.\n\n
#   With AI Agents, you can:\n\nAutomate lead qualification and prioritization\nIdentify and engage with high-potential customers\nStreamline your sales process and reduce
#   manual labor\n\nBy embracing AI Agents, you can:\n\nIncrease sales efficiency by up to 30%\nBoost conversion rates by up to 25%\nEnhance customer
#   relationships through personalized engagement\n\nThe future of B2B lead generation is here. Are you ready to revolutionize your sales approach?
#   Share your thoughts in the comments!\n\n**Source:** [Link to relevant thought leadership article or study]\n\n**Hashtags:** #AIAgent #SalesIntelligence
#   #B2BLeadGeneration #FutureOfWork #ArtificialIntelligence"\n\nFeel free to customize it to fit your brand\'s voice and style!'
#}