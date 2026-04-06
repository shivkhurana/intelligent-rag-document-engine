from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.vector_store import load_vector_db

app = FastAPI(title="RAG Document Engine API")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/search")
async def semantic_search(request: QueryRequest):
    try:
        # Load the vector database
        db = load_vector_db()
        
        # Perform similarity search
        results = db.similarity_search(request.query, k=3)
        
        # Extract the text from the top results
        context = [doc.page_content for doc in results]
        return {"status": "success", "retrieved_context": context}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))