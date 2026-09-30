import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME, PINECONE_API_KEY, EMBEDDING_MODEL, LLM_MODEL

app = FastAPI(
    title="Agentic AI RAG API",
    description="Strictly grounded RAG API powered by LangGraph, Pinecone, and Open-Source Models (Ollama)",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global graph instance (lazy loaded)
_graph_instance = None


def get_graph():
    """Retrieve or initialize compiled LangGraph workflow."""
    global _graph_instance
    if _graph_instance is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
        api_key = os.getenv("PINECONE_API_KEY", PINECONE_API_KEY)
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="PINECONE_API_KEY is not configured. Please add it to your .env file."
            )
        _graph_instance = build_rag_graph(index_name=index_name)
    return _graph_instance


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to be answered from the document")


class QueryResponse(BaseModel):
    final_answer: str = Field(..., description="Final grounded answer from the assistant")
    retrieved_context: List[str] = Field(..., description="Top context chunks retrieved from vector store")
    confidence_score: float = Field(..., description="Confidence or relevance score (0.0 to 1.0)")


@app.get("/")
async def root():
    return {
        "service": "Agentic AI RAG Chatbot API",
        "status": "online",
        "models": {
            "llm": LLM_MODEL,
            "embeddings": EMBEDDING_MODEL
        },
        "endpoints": {
            "/chat": "POST - Submit query to get grounded response",
            "/health": "GET - System health and configuration check",
            "/docs": "GET - Interactive Swagger UI"
        }
    }


@app.get("/health")
async def health_check():
    has_pinecone = bool(os.getenv("PINECONE_API_KEY", PINECONE_API_KEY))
    return {
        "status": "healthy" if has_pinecone else "warning",
        "pinecone_configured": has_pinecone,
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL
    }


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Execute stateful LangGraph RAG workflow:
    1. Retrieve relevant chunks from Pinecone
    2. Generate grounded answer strictly using context
    3. Return final answer, context chunks, and confidence score
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        graph = get_graph()
        initial_state = {
            "question": request.query.strip(),
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        result = graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result["answer"],
            retrieved_context=result["context"],
            confidence_score=result["score"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing RAG graph: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
