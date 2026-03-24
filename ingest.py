import os
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load credentials
load_dotenv()

def create_medical_index(pdf_path):
    # 2. Extract
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    # 3. Chunk (Smaller chunks = higher accuracy for medical facts)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    
    # 4. Embed & Store Locally (Free!)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This creates a folder named 'medical_db' to store the 'brain' of your bot
    vector_db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory="./medical_db"
    )
    print("Database created successfully!")

if __name__ == "__main__":
    # Replace with your actual PDF filename
    create_medical_index("WhereThereIsNoDoctor.pdf")