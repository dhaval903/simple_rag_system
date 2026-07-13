from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from duckduckgo_search import DDGS

from config import OPENAI_API_KEY

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

db=FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

def search_web(query):
    results = DDGS().text(query,max_results=5)
    
    text = ""
    
    for item in results:
        text += item['title'] + "\n"
        text += item['body'] + "\n\n"
        
    return text


def ask_question(question):
    
    docs = db.similarity_search_with_score(question,k=3)
    
    if docs and docs[0][1] > 0.5:
        context = "\n\n".join([doc[0].page_content for doc in docs])
        
        prompt = f"""
            Answer Only from document,
            
            Document: {context}
            
            Question : {question}
        """
        
        response = llm.invoke(prompt)
        
        return{
            "source":"Document",
            "answer":response.content
        }
        
    web_content = search_web(question)
    
    prompt = f"""

        Answer using web information.
        
        content: {web_content}
        
        Question:{question}
    
    """
    
    response= llm.invoke(prompt)
    
    return{
        "source":"Web Search",
        "answer":response.content
    }