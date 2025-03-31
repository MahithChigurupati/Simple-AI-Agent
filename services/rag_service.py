import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
import cohere
from langchain_cohere import CohereRerank
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

FAISS_INDEX_PATH = "faiss_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_PATH, "index.faiss")

vectorstore = None  # Don't load FAISS at import time

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        if os.path.exists(FAISS_INDEX_FILE):
            print("🔹 Loading existing FAISS index...")
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True)
        else:
            print("⚠️ FAISS index not found. Please generate embeddings first.")
            return None
    return vectorstore

# Define Retriever
top_k = 6
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k}) if vectorstore else None

# Use Cohere for Reranking (only if retriever is available)
if retriever:
    cohere_client = cohere.ClientV2(cohere_api_key)
    cohere_reranker = CohereRerank(client=cohere_client, model="rerank-english-v3.0", top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker,
        base_retriever=retriever
    )
else:
    compression_retriever = None

# Setup LLM (GPT-4) & Retrieval QA Chain
llm = ChatOpenAI(model="gpt-4o", temperature=0.5, max_tokens=3000)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=compression_retriever) if compression_retriever else None

# Function to query RAG system
def query_rag(user_query: str):
    if qa_chain is None:
        return {"error": "FAISS index not found. Please generate embeddings first."}
    response = qa_chain.invoke(user_query)
    return {"response": response["result"]}
