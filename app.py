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


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background: #f0f4f8;
    }

    /* ── Hide default Streamlit header/footer ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem;
        color: white;
    }
    .hero p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0;
    }

    /* ── Search card ── */
    .search-card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* ── Input field ── */
    .stTextInput > div > div > input {
        border: 2px solid #d1dce8;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 1rem;
        transition: border-color 0.2s;
        background: #f8fafc;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2d6a9f;
        box-shadow: 0 0 0 3px rgba(45,106,159,0.15);
        background: white;
    }

    /* ── Search button ── */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #1a3c5e, #2d6a9f);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2.2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.1s;
        width: 100%;
        margin-top: 0.5rem;
    }
    .stFormSubmitButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* ── Checkbox ── */
    .stCheckbox label {
        font-size: 0.95rem;
        color: #4a5568;
    }

    /* ── AI Answer box ── */
    .answer-box {
        background: linear-gradient(135deg, #eaf4ff, #f0f9f0);
        border-left: 4px solid #2d6a9f;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #1a202c;
    }

    /* ── Section heading ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Result card (expander) ── */
    .stExpander {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        overflow: hidden;
    }
    .stExpander summary {
        font-weight: 500;
        color: #2d3748;
        padding: 0.8rem 1rem;
    }
    .stExpander summary:hover {
        background: #f7fafc;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #1a3c5e;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #2d5a8a;
        border: 1px solid #4a7baa;
        border-radius: 8px;
        color: white;
    }

    /* ── Warning / info messages ── */
    .stWarning {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False)
def get_book_titles(_vectorstore) -> list[str]:
    result = _vectorstore.get(include=["metadatas"])
    titles = sorted({
        m.get("book_title", "Unknown")
        for m in result["metadatas"]
        if m.get("book_title")
    })
    return titles


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

inject_css()

vectorstore = load_vectorstore()
llm = load_llm()

# Language selector in sidebar
with st.sidebar:
    st.markdown("### 🌐 Language / Мова / Jezik")
    lang_name = st.selectbox(
        "",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**📚 Books in the database**")
    st.markdown("""
- 101 Ways to Create New Income Streams
- Personal Finance for Those Who Want to Get Everything Done — *Ilina*
- The Next Millionaire Next Door — *Stanley & Fallaw*
- The Little Book of Common Sense Investing — *John C. Bogle*
- Rich Dad Poor Dad — *Robert T. Kiyosaki*
- Girl with Money
- Your Money or Your Life — *Robin & Domínguez*
- Fundamentals of Financial Literacy: A Short Course
- Rules of a Rich Grandmother
- Smart Insurance — *S. Biryukov*
""")

lang_code = LANGUAGE_OPTIONS[lang_name]
t = TRANSLATIONS[lang_code]

# Hero banner
st.markdown(f"""
<div class="hero">
    <h1>{t["app_title"]}</h1>
    <p>{t["app_subtitle"]}</p>
</div>
""", unsafe_allow_html=True)

# Search form
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

            st.markdown(f'<div class="section-title">🤖 {t["header_answer"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True)

        st.markdown(
            f'<div class="section-title">📖 {t["header_results"].format(n=len(results))}</div>',
            unsafe_allow_html=True,
        )
        for i, doc in enumerate(results, start=1):
            book = doc.metadata.get("book_title", "Unknown book")
            page = doc.metadata.get("page_number", "?")
            with st.expander(f"{i}. {book} — {t['page_label']} {page}"):
                st.write(doc.page_content)

elif submitted:
    st.warning(t["warn_empty_query"])
