from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Summarization App")

st.header("Summarization App")

hfLLM = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chatModel = ChatHuggingFace(llm = hfLLM)

paper_input = st.selectbox(
    label="Select research paper name",
    options=[
        "Attension all you need",
        "BERT: Pre-training of deep BiDirectional transformers",
        "GPT3: Language model are Few-Shot learner",
        "Diffusion model beats GANs of Image Synthesis"
    ]
)

style_input = st.selectbox(
    label="Select explanation style",
    options=[
        "Beginner-Friendly",
        "Technical code-oriented",
        "Matemetical"
    ]
)

length_input = st.selectbox(
    label="Select research paper name",
    options=[
        "Short [1-2 para]",
        "Medium [3-5 para]",
        "Long [Detailed explanation]"
    ]
)

template = PromptTemplate(
    template="""Please summarize the research paper titled "{paper_input}" with the following
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets
where applicable.
2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient
information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and
length.""",
input_variables=['paper_input', 'style_input', 'length_input']
)

# Fill the placeholders
prompt = template.invoke(
    {
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input 
    }
)

# Button
if st.button("Summarize"):
    if paper_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        # Placeholder for summarization logic
        st.success("Summarization will appear here...")
        result = chatModel.invoke(prompt)
        st.write(result.content)  # for now just showing input
        