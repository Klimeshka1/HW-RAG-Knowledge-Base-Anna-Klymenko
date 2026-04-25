"""
app.py — Streamlit UI for Financial Literacy RAG app.

Supports English, Ukrainian, and Slovenian UI languages.
Run: streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, SystemMessage

from translations import TRANSLATIONS, LANGUAGE_OPTIONS

load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_literacy"
TOP_K = 5


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file or Render environment.")
        st.stop()

    if not CHROMA_DIR.exists():
        st.error(
            f"ChromaDB not found at `{CHROMA_DIR}`. "
            "Run `python ingest.py` first to process your PDF books."
        )
        st.stop()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key,
    )


def search(vectorstore, query: str) -> list:
    return vectorstore.similarity_search(query, k=TOP_K)


def generate_answer(llm, query: str, chunks: list, system_prompt: str) -> str:
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('book_title', 'Unknown')} — p.{doc.metadata.get('page_number', '?')}]\n"
        f"{doc.page_content}"
        for doc in chunks
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    response = llm.invoke(messages)
    return response.content


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Literacy Search",
    page_icon="📚",
    layout="centered",
)

# Language selector in sidebar
with st.sidebar:
    lang_name = st.selectbox(
        "🌐 Language / Мова / Jezik",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
    )

lang_code = LANGUAGE_OPTIONS[lang_name]
t = TRANSLATIONS[lang_code]

st.title(t["app_title"])
st.caption(t["app_subtitle"])

vectorstore = load_vectorstore()
llm = load_llm()

with st.form("search_form"):
    query = st.text_input(
        t["input_label"],
        placeholder=t["input_placeholder"],
    )
    use_llm = st.checkbox(t["checkbox_llm"], value=True)
    submitted = st.form_submit_button(t["button_search"])

if submitted and query.strip():
    with st.spinner(t["spinner_search"]):
        results = search(vectorstore, query)

    if not results:
        st.warning(t["warn_no_results"])
    else:
        if use_llm:
            with st.spinner(t["spinner_answer"]):
                answer = generate_answer(llm, query, results, t["system_prompt"])
            st.subheader(t["header_answer"])
            st.write(answer)
            st.divider()

        st.subheader(t["header_results"].format(n=len(results)))
        for i, doc in enumerate(results, start=1):
            book = doc.metadata.get("book_title", "Unknown book")
            page = doc.metadata.get("page_number", "?")
            with st.expander(f"{i}. {book} — {t['page_label']} {page}"):
                st.write(doc.page_content)

elif submitted:
    st.warning(t["warn_empty_query"])
