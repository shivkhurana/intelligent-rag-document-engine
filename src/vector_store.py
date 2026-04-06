from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_db(chunks):
    # Use an open-source Transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store high-dimensional embeddings in FAISS
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings)