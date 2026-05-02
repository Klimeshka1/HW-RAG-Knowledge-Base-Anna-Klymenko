"""
app.py — Streamlit UI for Financial Literacy RAG app.

Pages: Search, Statistics, About.
Supports English, Ukrainian, and Slovenian UI languages.
Run: streamlit run app.py
"""

import os
from pathlib import Path
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
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

# Clean display names for each internal ChromaDB title
BOOK_DISPLAY_NAMES = {
    "101 Способ Создания Новых Источников Дохода. Как Зарабатывать На Всем И Всегда (Epub)   Флибуста": "101 Ways to Create New Income Streams",
    "Ilina Lichnye Finansy Dlya Teh Kto Hochet Vse Uspet.626332": "Personal Finance — Ilina",
    "Thomas J. Stanley  Sarah Stanley Fallaw   The Next Millionaire Next Door  Enduring Strategies For Building Wealth (2018, Lyons Press)   Libgen.Li": "The Next Millionaire Next Door",
    "[Little Book Big Profits] John C Bogle   The Little Book Of Common Sense Investing   The Only Way To Guarantee Your Fair Share Of Market Returns (2007, John Wiley & Sons )   Libgen.Li": "The Little Book of Common Sense Investing",
    "Богатый Папа, Бедный Папа (Fb2)   Флибуста": "Rich Dad Poor Dad",
    "Девушка С Деньгами (Fb2)   Флибуста": "Girl with Money",
    "Кошелек Или Жизнь  (Fb2)   Флибуста": "Your Money or Your Life",
    "Основы Финансовой Грамотности  Краткий Курс (Fb2)   Флибуста": "Fundamentals of Financial Literacy",
    "Правила Богатой Бабушки. Финансовая Грамотность Для Жизни Вашей Мечты (Fb2)   Флибуста": "Rules of a Rich Grandmother",
    "Разумное Страхование. Актуальные Рыночные Практики{Бирюков С.}{111952649} Libgen.Li": "Smart Insurance — Biryukov",
}

BOOK_LANGUAGES = {
    "101 Ways to Create New Income Streams": "Russian",
    "Personal Finance — Ilina": "Russian",
    "The Next Millionaire Next Door": "English",
    "The Little Book of Common Sense Investing": "English",
    "Rich Dad Poor Dad": "Russian",
    "Girl with Money": "Russian",
    "Your Money or Your Life": "Russian",
    "Fundamentals of Financial Literacy": "Russian",
    "Rules of a Rich Grandmother": "Russian",
    "Smart Insurance — Biryukov": "Russian",
}


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #f0f4f8; }

    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem; color: white; }
    .hero p { font-size: 1rem; opacity: 0.85; margin: 0; }

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
    .stFormSubmitButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

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

    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #2d3748;
        margin: 1.5rem 0 0.8rem;
        display: flex; align-items: center; gap: 0.4rem;
    }

    .stExpander {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; margin-bottom: 0.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05); overflow: hidden;
    }

    .stat-card {
        background: white; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .stat-number { font-size: 2.2rem; font-weight: 700; color: #2d6a9f; }
    .stat-label { font-size: 0.9rem; color: #718096; margin-top: 0.2rem; }

    section[data-testid="stSidebar"] { background: #1a3c5e; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #2d5a8a; border: 1px solid #4a7baa; border-radius: 8px;
    }
    section[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set.")
        st.stop()
    if not CHROMA_DIR.exists():
        st.error("ChromaDB not found. Run `python ingest.py` first.")
        st.stop()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


@st.cache_data(show_spinner=False)
def get_stats(_vectorstore):
    result = _vectorstore.get(include=["metadatas"])
    chunk_counts = Counter(
        m.get("book_title", "Unknown") for m in result["metadatas"]
    )
    return dict(chunk_counts), len(result["metadatas"])


def search(vectorstore, query):
    return vectorstore.similarity_search(query, k=TOP_K)


def generate_answer(llm, query, chunks, system_prompt):
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('book_title', 'Unknown')} — p.{doc.metadata.get('page_number', '?')}]\n"
        f"{doc.page_content}"
        for doc in chunks
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    return response.content


# ── App setup ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Financial Literacy Search", page_icon="📚", layout="centered")
inject_css()

vectorstore = load_vectorstore()
llm = load_llm()

with st.sidebar:
    st.markdown("### 📚 Financial Literacy")
    page = st.radio(
        "Navigate",
        ["🔍 Search", "📊 Statistics", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 🌐 Language / Мова / Jezik")
    lang_name = st.selectbox("", list(LANGUAGE_OPTIONS.keys()), label_visibility="collapsed")

lang_code = LANGUAGE_OPTIONS[lang_name]
t = TRANSLATIONS[lang_code]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>{t["app_title"]}</h1>
    <p>{t["app_subtitle"]}</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Search":
    with st.form("search_form"):
        query = st.text_input(t["input_label"], placeholder=t["input_placeholder"])
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
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="section-title">📖 {t["header_results"].format(n=len(results))}</div>',
                unsafe_allow_html=True,
            )
            for i, doc in enumerate(results, start=1):
                book = doc.metadata.get("book_title", "Unknown book")
                display = BOOK_DISPLAY_NAMES.get(book, book)
                page_num = doc.metadata.get("page_number", "?")
                with st.expander(f"{i}. {display} — {t['page_label']} {page_num}"):
                    st.write(doc.page_content)

    elif submitted:
        st.warning(t["warn_empty_query"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistics":
    chunk_counts, total_chunks = get_stats(vectorstore)

    # Map to clean display names
    display_counts = {
        BOOK_DISPLAY_NAMES.get(k, k): v for k, v in chunk_counts.items()
    }

    # ── Top metrics ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">10</div><div class="stat-label">Books</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{total_chunks:,}</div><div class="stat-label">Total chunks</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">3</div><div class="stat-label">UI languages</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart: chunks per book ──
    st.subheader("Chunks per book")
    books = list(display_counts.keys())
    counts = list(display_counts.values())
    fig_bar = px.bar(
        x=counts, y=books,
        orientation="h",
        color=counts,
        color_continuous_scale=["#a8c8e8", "#2d6a9f", "#1a3c5e"],
        labels={"x": "Chunks", "y": ""},
        text=counts,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=10, r=30, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Pie chart: language distribution ──
    st.subheader("Language distribution")
    lang_counts = Counter(
        BOOK_LANGUAGES.get(name, "Unknown") for name in display_counts.keys()
    )
    fig_pie = px.pie(
        names=list(lang_counts.keys()),
        values=list(lang_counts.values()),
        color_discrete_sequence=["#2d6a9f", "#48bb78"],
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="label+percent", textfont_size=14)
    fig_pie.update_layout(
        height=350,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("## About this app")
    st.markdown("""
This application is a **RAG (Retrieval-Augmented Generation)** knowledge base
built on a collection of 10 financial literacy books in Russian and English.

Type a question in natural language — the app finds the most relevant passages
from the books using semantic search, and optionally generates a concise answer
using GPT-4o-mini.
    """)

    st.markdown("## Tech stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| Component | Technology |
|-----------|-----------|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector DB | ChromaDB (persistent, local) |
| LLM | GPT-4o-mini |
| Framework | LangChain |
        """)
    with col2:
        st.markdown("""
| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Languages | English · Ukrainian · Slovenian |
| Deployment | Render.com |
| Source control | GitHub |
        """)

    st.markdown("## Links")
    st.markdown("""
- 🐙 **GitHub:** https://github.com/Klimeshka1/HW-RAG-Knowledge-Base-Anna-Klymenko
- 🌐 **Live app:** https://hw-rag-knowledge-base-anna-klymenko.onrender.com
    """)

    st.markdown("## Author")
    st.markdown("**Anna Klymenko** — university project, 2026")
