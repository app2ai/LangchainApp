from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

os.environ["HF_HOME"] = "D:/HF AI Models"

indian_cricket_sentences = [
    "The Indian cricket team has been shaped by legends like MS Dhoni, Virat Kohli, Rohit Sharma, and Sachin Tendulkar.",
    "Sachin Tendulkar, known as the 'God of Cricket,' inspired an entire generation of players in India.",
    "MS Dhoni is celebrated for his calm leadership and led India to victory in the 2011 World Cup.",
    "Virat Kohli is known for his aggressive batting style and exceptional consistency across all formats.",
    "Rohit Sharma, also called the 'Hitman,' holds the record for multiple double centuries in ODIs.",
    "Under Dhonis captaincy, India became the number one Test team and won multiple ICC trophies.",
    "Virat Kohli took Indian fitness standards to a new level during his captaincy tenure.",
    "Rohit Sharma has been a successful captain, especially in limited-overs cricket.",
    "Sachin Tendulkars 100 international centuries remain a historic milestone in cricket.",
    "Together, these players represent different eras of dominance and pride in Indian cricket history."
]

query = "Who is best captain?"

eModel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

docEmbeddings = eModel.embed_documents(indian_cricket_sentences)
queryEmbedding = eModel.embed_query(query)

similarity = cosine_similarity([queryEmbedding], docEmbeddings)[0]

maxPossibleAns = sorted(list(enumerate(similarity)), key= lambda x:x[1]) # sorted list

index, score = maxPossibleAns[-1] # Last item of list

print(query)
print(indian_cricket_sentences[index])
print("Score for similarity: ", score)
