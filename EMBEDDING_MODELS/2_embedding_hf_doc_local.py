from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ["HF_HOME"] = "D:/HF AI Models"

doc = ["Delhi is capital of India", "Modi is PM of India", "India is Hindu Nation."]

eModel = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vecters = eModel.embed_documents(doc)

print(str(vecters))
