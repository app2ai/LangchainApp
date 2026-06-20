from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import os

os.environ["HF_HOME"] = "D:/HF AI Models"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is capital of Austrelia?")

print(result.content)