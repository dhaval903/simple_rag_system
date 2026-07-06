from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from duckduckgo_search import DGS

from config import openai_api_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db=FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_api_key
)

def search_web(query):
    results = DGS().text(query,max_results=5)
    
    text = ""
    
    for item in results:
        text += item['title'] + "\n"
        text += item['body'] + "\n\n"
        
    return text


def ask_question(question):
    
    docs = db.similarity_search_with_score(question,k=3)
    
    if docs and docs[0][1] > 0.5:
        context = "\n\n".join([doc[0].page_content for doc in docs])