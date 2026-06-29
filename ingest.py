import os

from config import OPENAI_API_KEY

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "documents/medical_report.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size =  1000,
    chunk_overlap = 200
)

docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    openai_api_key = OPENAI_API_KEY
)

db = FAISS.from_documents(
    docs,
    embeddings
)

db.save_local("vectorstore")

print("vector db created")