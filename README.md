# Financial Literacy Knowledge Base

A RAG (Retrieval-Augmented Generation) app for semantic search over financial literacy books.

Built with LangChain · ChromaDB · OpenAI · Streamlit.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd HW-RAG-Knowledge-Base-Anna-Klymenko
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and replace "your_key_here" with your actual key
```

### 3. Add PDF books

Place your PDF files in `data/books/`. The filename becomes the book title
(underscores and dashes are replaced with spaces).

```
data/books/
├── Rich_Dad_Poor_Dad.pdf
├── The_Intelligent_Investor.pdf
└── ...
```

### 4. Run ingestion

```bash
python ingest.py
```

To wipe the database and start fresh:

```bash
python ingest.py --reset
```

This creates a `chroma_db/` folder with all embeddings. **Commit this folder to Git**
so the app works on Render without re-ingesting.

### 5. Run the app locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Deploying to Render.com

1. Push the repo (including `chroma_db/`) to GitHub.
2. Create a new **Web Service** on Render, connect your GitHub repo.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add environment variable: `OPENAI_API_KEY` = your key.
6. Deploy.

---

## Project structure

```
├── data/books/        # PDF source documents (not committed)
├── chroma_db/         # Persistent vector store (committed after ingestion)
├── ingest.py          # PDF → chunks → embeddings → ChromaDB
├── app.py             # Streamlit web UI
├── requirements.txt
├── .env.example
└── README.md
```
