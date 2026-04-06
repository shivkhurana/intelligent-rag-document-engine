from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_document(file_path: str):
    # Load the unstructured text
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Chunk the text to fit into LLM context windows
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks