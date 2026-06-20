from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedModel = VertexAIEmbeddings(model="textembedding-gecko@003", dimensions=16)

query = "Narendra Damodardas Modi is PM of republic of India."

vectors = embedModel.embed_query(query)

print(vectors)
