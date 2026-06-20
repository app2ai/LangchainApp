from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Summarization App")

st.header("📝 Summarization App")

# Text input area
user_input = st.text_area("Enter your text here:", height=200)

hfLLM = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chatModel = ChatHuggingFace(llm = hfLLM)

# Button
if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        # Placeholder for summarization logic
        st.success("Summarization will appear here...")
        result = chatModel.invoke(user_input)
        st.write(result.content)  # for now just showing input

