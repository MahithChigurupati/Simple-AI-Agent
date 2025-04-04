import os
import fitz
import glob
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Define content directory (parent dir of project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
CONTENT_DIR = os.path.join(BASE_DIR, "content")

def load_all_pdfs_in_directory(directory=CONTENT_DIR):
    combined_text = ""
    pdf_paths = glob.glob(f"{directory}/*.pdf")

    for file_path in pdf_paths:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                combined_text += page.get_text()

    return combined_text if combined_text else None

def process_pdfs_and_create_embeddings():
    pdf_data = load_all_pdfs_in_directory()
    
    if not pdf_data:
        return {"message": "No PDFs found or PDFs contain no extractable text."}

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_text(pdf_data)

    # Generate embeddings
    embedded_documents = embeddings_model.embed_documents(documents)

    # Store embeddings in FAISS
    faiss_index = FAISS.from_texts(texts=documents, embedding=embeddings_model)
    faiss_index.save_local(os.path.join(BASE_DIR, "faiss_index"))

    return {
        "message": "Embeddings created successfully",
        "total_documents": len(documents)
    }
