import os
import sys
from typing import List, TypedDict, Optional
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Ensure parent directory is in path when executed directly
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.config import (
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    TOP_K,
)


class AgentState(TypedDict):
    """
    State dictionary passed through the LangGraph execution graph.
    - question: The user query string
    - context: List of retrieved context text chunks from the vector database
    - answer: The final generated response from the LLM
    - score: Confidence / relevance score reflecting grounding quality
    """
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph(index_name: str = PINECONE_INDEX_NAME, vectorstore=None):
    """
    Constructs and compiles the LangGraph stateful execution graph:
    START -> retrieve -> generate -> END
    Uses open-source Ollama embeddings (nomic-embed-text) and open-source LLM (llama3.2).
    """
    # Open-source embeddings
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Initialize Pinecone vector store if not explicitly injected
    if vectorstore is None:
        if not PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is not set in environment or .env file. "
                "Please configure PINECONE_API_KEY to query the vector index."
            )
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Open-source LLM
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

    # Define Node 1: Retrieve
    def retrieve_node(state: AgentState):
        question = state["question"]
        docs = retriever.invoke(question)
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    # Define Node 2: Generate (Strict Grounding)
    def generate_node(state: AgentState):
        context_texts = state.get("context", [])
        if not context_texts:
            return {
                "answer": "I cannot answer based on the provided document.",
                "score": 0.0,
            }

        context_str = "\n\n".join(context_texts)
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
If the context does not contain enough info, state 'I cannot answer based on the provided document.'

Context:
{context_str}

Question: {state['question']}
Answer:"""

        response = llm.invoke(prompt)
        answer_text = response.content.strip()

        # Score computation: check for explicit refusal or lack of context
        refusal_markers = [
            "cannot answer based on the provided document",
            "not mentioned in the provided",
            "does not contain information",
            "not found in the provided context",
            "no mention of",
        ]
        is_refusal = any(marker in answer_text.lower() for marker in refusal_markers)

        if is_refusal:
            confidence = 0.0
        elif len(context_texts) > 0:
            confidence = 0.95
        else:
            confidence = 0.0

        return {"answer": answer_text, "score": confidence}

    # Assemble LangGraph StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
