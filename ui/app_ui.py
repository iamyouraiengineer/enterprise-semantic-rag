"""
Streamlit frontend for the Enterprise Semantic RAG Engine.

Design decisions:
- The UI is a thin client that calls the FastAPI backend via HTTP.
  This ensures the UI and API can run independently (e.g., UI on port 8501,
  API on port 8000) and matches production microservice patterns.
- Document upload is in the sidebar with progress feedback.
- Chat history is stored in Streamlit session_state so it persists
  across rerenders.
- Source citations are displayed in expandable cards with metadata
  (source file, chunk index, text preview).
- We use st.spinner during API calls to give the user visual feedback.
"""

import os

import streamlit as st
import httpx

# =============================================================================
# Configuration
# =============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Helper Functions
# =============================================================================
def call_api(endpoint: str, payload: dict) -> dict:
    """
    Make a POST request to the FastAPI backend.
    Returns the JSON response or an error dict.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def render_source_card(source: dict, index: int) -> None:
    """Render a single source citation in an expander."""
    with st.expander(f"📄 Source {index + 1}: {source['source']} (chunk {source['chunk_index']})"):
        st.markdown(f"```\n{source['text_preview']}\n```")


# =============================================================================
# Sidebar: Document Upload
# =============================================================================
with st.sidebar:
    st.title("📁 Document Upload")
    st.markdown("Upload documents to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose files (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("🚀 Ingest Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Read file contents
                documents = []
                for file in uploaded_files:
                    content = file.read().decode("utf-8", errors="ignore")
                    documents.append({
                        "text": content,
                        "source": file.name,
                        "metadata": {"type": file.name.split(".")[-1]},
                    })

                # Call API
                result = call_api("/ingest", {"documents": documents})

                if "error" in result:
                    st.error(f"Ingestion failed: {result['error']}")
                else:
                    st.success(
                        f"✅ Ingested {result['documents_ingested']} documents "
                        f"into {result['chunks_created']} chunks"
                    )

    st.divider()
    st.markdown("---")
    st.caption(f"API: {API_BASE_URL}")


# =============================================================================
# Main: Chat Interface
# =============================================================================
st.title("🔍 Enterprise Semantic Search & RAG")
st.markdown("Ask questions about your uploaded documents. Answers include source citations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.divider()
            st.markdown("**Sources:**")
            for i, source in enumerate(msg["sources"]):
                render_source_card(source, i)
            if msg.get("latency_ms"):
                st.caption(f"⏱️ Latency: {msg['latency_ms']}ms | Tokens: {msg.get('token_usage', 0)}")

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = call_api("/query", {
                "question": question,
                "top_k": 5,
                "rerank": True,
            })

            if result.get("error"):
                answer = f"❌ Error: {result['error']}"
                sources = []
                latency = 0
                tokens = 0
            else:
                answer = result["answer"]
                sources = result.get("sources", [])
                latency = result.get("latency_ms", 0)
                tokens = result.get("token_usage", 0)

            # Display answer
            st.markdown(answer)

            # Display sources
            if sources:
                st.divider()
                st.markdown("**Sources:**")
                for i, source in enumerate(sources):
                    render_source_card(source, i)
                st.caption(f"⏱️ Latency: {latency}ms | Tokens: {tokens}")

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "latency_ms": latency,
        "token_usage": tokens,
    })
