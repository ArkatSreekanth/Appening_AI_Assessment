import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
)
from src.graph import build_rag_graph

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI Chatbot")
st.caption("Grounded RAG with LangGraph, Pinecone & Open-Source LLMs (Ollama)")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration & Status")
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Embedding:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Index:** `{PINECONE_INDEX_NAME}`")

    has_pinecone = bool(os.getenv("PINECONE_API_KEY", PINECONE_API_KEY))
    if has_pinecone:
        st.success("Pinecone API Key Configured ✅")
    else:
        st.error("Pinecone API Key Missing ❌")
        st.info("Add `PINECONE_API_KEY=...` to your `.env` file.")

    st.divider()

    st.subheader("💡 Benchmark Sample Queries")
    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]
    for q in sample_queries:
        if st.button(q, key=f"btn_{q[:15]}", use_container_width=True):
            st.session_state["preset_query"] = q

    st.divider()
    st.subheader("📚 Retrieved Context & Score")
    last_context = st.session_state.get("last_context", [])
    last_score = st.session_state.get("last_score", None)

    if last_score is not None:
        if last_score >= 0.8:
            st.success(f"Confidence Score: {last_score:.2f} (High Grounding)")
        elif last_score > 0.0:
            st.warning(f"Confidence Score: {last_score:.2f} (Moderate)")
        else:
            st.error(f"Confidence Score: {last_score:.2f} (No context / Refusal)")

    if last_context:
        st.markdown(f"**Retrieved Chunks ({len(last_context)}):**")
        for i, chunk in enumerate(last_context):
            with st.expander(f"Chunk {i+1}", expanded=(i == 0)):
                st.write(chunk)
    else:
        st.caption("Ask a question to view retrieved context chunks.")

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score" in msg and msg["score"] is not None:
            st.caption(f"Confidence: {msg['score']:.2f}")

# Handle preset question button click
query_to_submit = None
if "preset_query" in st.session_state:
    query_to_submit = st.session_state.pop("preset_query")

user_input = st.chat_input("Ask a question based strictly on the Agentic AI eBook...")
if user_input:
    query_to_submit = user_input

if query_to_submit:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": query_to_submit})
    with st.chat_message("user"):
        st.markdown(query_to_submit)

    # Invoke pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving chunks and reasoning strictly over context..."):
            try:
                # Call LangGraph pipeline directly
                graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
                result = graph.invoke({
                    "question": query_to_submit,
                    "context": [],
                    "answer": "",
                    "score": 0.0,
                })

                answer = result["answer"]
                context_chunks = result["context"]
                confidence_score = result["score"]

                st.markdown(answer)
                st.caption(f"Grounding Confidence Score: **{confidence_score:.2f}**")

                # Update sidebar state
                st.session_state["last_context"] = context_chunks
                st.session_state["last_score"] = confidence_score

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "score": confidence_score,
                    "context": context_chunks,
                })

            except Exception as e:
                st.error(f"Error executing RAG pipeline: {str(e)}")
