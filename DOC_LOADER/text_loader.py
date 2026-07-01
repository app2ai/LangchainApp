from langchain_community.document_loaders import TextLoader, Docling
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

docsData = TextLoader("./DOC_LOADER/files/normal_text.txt").load()

print(docsData[0].page_content)
print(docsData[0].metadata)
