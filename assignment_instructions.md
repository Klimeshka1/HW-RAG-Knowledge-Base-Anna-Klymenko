# Assignment: Build Your Own RAG Knowledge Base

## Overview

In this assignment, you will build a **Retrieval-Augmented Generation (RAG) web application** that allows users to search through a collection of documents about a topic of your choice using **semantic search**. You will then deploy it live on **Render.com** so anyone can access it.

This assignment ties together everything you have learned over the past three days: text processing, embeddings, vector databases, web applications, and deployment.

---

## Learning Objectives

By completing this assignment, you will:

- Apply chunking strategies to prepare documents for embedding
- Use **LangChain** and **ChromaDB** to build a vector database
- Build an interactive web application with **Streamlit**
- Deploy a real application using **Git**, **GitHub**, and **Render.com**
- Document and present your work professionally

---

## What You Will Build

A **Streamlit web application** with:

- At least **10 documents/texts** about your chosen topic
- **Semantic search functionality** (user types a question, app returns relevant chunks)
- At least **2 pages** (Home + Search, optionally more)
- A clean, user-friendly interface

---

## Topic Ideas

Pick a topic you find interesting! Here are some ideas to get you started:

### Academic
1. A specific university course (e.g., Introduction to Psychology, Linear Algebra)
2. A scientific field (e.g., quantum physics, marine biology, neuroscience)
3. Historical events (e.g., the Space Race, the Renaissance, ancient civilizations)
4. Famous scientists and their discoveries

### Hobbies & Lifestyle
5. Recipes from a specific cuisine (e.g., Italian, Japanese, Mexican)
6. Rules and strategies for a sport (e.g., basketball, chess, climbing)
7. Video game guides or lore (e.g., Zelda, Minecraft, League of Legends)
8. Music theory fundamentals (scales, chords, rhythm)
9. Fitness exercises and workout plans

### Professional & Technical
10. A programming language or framework (e.g., Python, React, SQL)
11. Design principles (UI/UX, graphic design, typography)
12. Business and startup concepts (lean methodology, marketing, finance)
13. Cybersecurity fundamentals

### Fun & Creative
14. Movie plots or film analysis (e.g., Studio Ghibli, Marvel, Nolan films)
15. Book summaries from your favorite genre
16. Travel guides for countries or cities you love
17. Animal facts (ocean creatures, endangered species, dinosaurs)
18. Mythology and folklore from different cultures

### Personal
19. FAQ about your hometown or country
20. Your own CV/portfolio (turn yourself into a searchable knowledge base!)
21. Your study notes from a course you are taking

Feel free to come up with your own topic -- the more passionate you are about it, the better your project will be!

---

## Step-by-Step Instructions

### Step 1: Choose Your Topic (5 min)

Pick something you are genuinely interested in. You will need to find or write at least **10 text passages** about it. The topic should be broad enough that you can cover different aspects of it.

**Ask yourself**: Can I write (or find) 10 different paragraphs about different subtopics within this theme?

---

### Step 2: Collect Your Documents (30 min)

Gather your source material:

- **Write your own text** OR **copy from Wikipedia, textbooks, articles, etc.**
- Each document should be **1-3 paragraphs** long (roughly **100-500 words**)
- Aim for at least **10 documents** covering **different aspects** of your topic
- Save them as the `DOCUMENTS` list in your `app.py` file

**Tips for good documents:**
- Cover different subtopics (breadth is better than depth for search)
- Include specific facts, names, dates, or numbers where relevant
- Write in clear, informative language
- If copying from the web, make sure to note your sources for the report

**Example**: If your topic is "Italian Cuisine," your documents might cover: pasta types, pizza history, regional differences, olive oil, famous chefs, wine pairing, desserts, cooking techniques, Italian food culture, and a famous recipe.

---

### Step 3: Set Up Your Project (10 min)

Create a project folder with the following files:

```
my-rag-app/
    app.py                 # Your Streamlit application
    requirements.txt       # Python dependencies
    render.yaml            # Render deployment configuration
    .gitignore             # Files to exclude from Git
```

**`requirements.txt`** should contain:
```
streamlit
langchain
langchain-community
langchain-text-splitters
chromadb
sentence-transformers
```

**`render.yaml`** should contain:
```yaml
services:
  - type: web
    name: my-rag-app
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
```

**`.gitignore`** should contain:
```
__pycache__/
*.pyc
.env
chroma_db/
```

For `app.py`, start from the **starter template** provided in the course materials (see the `streamlit-starter` folder).

---

### Step 4: Customize Your App (60 min)

Starting from the starter template, make the following changes:

1. **Replace the `DOCUMENTS` list** with your own texts (at least 10)
2. **Update the app title, icon, and descriptions** to match your topic
3. **Choose a chunking strategy** and set your `chunk_size` and `chunk_overlap`
   - Think about what makes sense for your documents
   - Shorter chunks (100-200 chars) = more precise but less context
   - Longer chunks (500-1000 chars) = more context but less precise
   - You will need to explain your choice in the report!
4. **(Optional)** Add extra pages: About, Statistics, Visualization
5. **(Optional)** Customize the styling with colors, layout, or images

**Important**: Test different chunking strategies! Try at least two different `chunk_size` values and see how the search results change. This will give you something interesting to write about in your report.

---

### Step 5: Test Locally (15 min)

Run your application locally to make sure everything works:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your app should open in the browser at `http://localhost:8501`.

**Test checklist:**
- [ ] App loads without errors
- [ ] Home page displays correctly with your topic information
- [ ] Search page accepts a query and returns results
- [ ] Results are relevant to the query
- [ ] Try at least **5 different search queries**
- [ ] Try a query that is NOT about your topic -- what happens?

Take a **screenshot** of your working app for the report.

---

### Step 6: Push to GitHub (10 min)

First, create a new repository on GitHub:
1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** button (top right) and select **"New repository"**
3. Name it something descriptive (e.g., `italian-cuisine-rag`)
4. Set it to **Public**
5. Do NOT initialize with a README (we will push from local)
6. Click **"Create repository"**

Then, push your code from the terminal:

```bash
git init
git add app.py requirements.txt render.yaml .gitignore
git commit -m "My RAG knowledge base app"
git remote add origin https://github.com/YOUR_USERNAME/your-repo-name.git
git branch -M main
git push -u origin main
```

**Replace** `YOUR_USERNAME` and `your-repo-name` with your actual GitHub username and repository name.

---

### Step 7: Deploy to Render (15 min)

1. Go to [render.com](https://render.com) and sign in with your **GitHub account**
2. From the dashboard, click **"New +"** and select **"Web Service"**
3. Find and connect your repository from the list
4. Render should automatically detect your `render.yaml` configuration
5. Click **"Create Web Service"**
6. Wait for the build to complete (this may take 5-10 minutes)
7. Once deployed, click the URL at the top of the page to see your live app
8. Test your live URL to make sure everything works

**Save your Render URL** -- you will need it for the report!

> **Note**: The free Render tier "sleeps" your app after 15 minutes of inactivity. The first visit after sleep takes about 30-60 seconds to wake up. This is normal.

---

### Step 8: Write Your Report (45 min)

See the Report Requirements section below for details.

---

## Report Requirements

Submit a **PDF report** (2-3 pages) containing the following:

### Page 1: Application Overview

- **Title**: Your app name and topic
- **Topic description**: Why you chose this topic (2-3 sentences)
- **Screenshot**: A screenshot of your running application showing the search page with results
- **Links**:
  - GitHub repository URL
  - Render deployment URL

### Page 2: Technical Details

- **Documents**: How many documents you included and where you sourced them (self-written, Wikipedia, textbooks, etc.)
- **Chunking strategy**: This is the most important part!
  - What `chunk_size` and `chunk_overlap` did you use?
  - **Why** did you choose those values?
  - Did you try multiple strategies? What differences did you notice?
  - Give a concrete example: "When I searched for X, with chunk_size=200 I got Y, but with chunk_size=500 I got Z"
- **Embedding model**: Which model you used (default: `all-MiniLM-L6-v2`) and any notes about it
- **Interesting findings**: Any surprising search results? Queries that worked unexpectedly well or poorly? Edge cases you discovered?

### Page 3 (Optional): Reflections & Extensions

- What did you learn from this project?
- What would you improve if you had more time?
- Any ideas for extending the app (e.g., adding an LLM for answer generation, supporting file uploads, adding more documents)?

---

## Grading Rubric

| Criterion | Points | Description |
|---|---|---|
| **Working Application** | 40 | App runs locally AND on Render. Search returns relevant results for at least 3 different queries. |
| **Code Quality** | 20 | Clean, readable code. Sensible chunking strategy with justification. At least 10 documents included. |
| **Deployment** | 20 | App is live and accessible on Render.com. GitHub repo is public and contains all required files (`app.py`, `requirements.txt`, `render.yaml`, `.gitignore`). |
| **Report** | 20 | Complete report with screenshots, working links, and a clear technical explanation of your chunking choice. |
| **Total** | **100** | |

### Bonus Points (up to +10)

| Bonus | Points | Description |
|---|---|---|
| Additional pages | +3 | Added pages beyond Home and Search (e.g., About, Statistics, Visualization) |
| Custom styling | +3 | Creative UI design, custom colors, images, or layout |
| Chunking comparison | +2 | Tried multiple chunking strategies and compared them in the report with examples |
| Multilingual | +2 | Used a non-English language with an appropriate multilingual embedding model |

---

## Submission

- **Deadline**: [INSTRUCTOR: SET DEADLINE HERE]
- **What to submit**:
  1. PDF report (uploaded to the course platform)
  2. The report **must** contain your GitHub repo URL and Render URL
- **Important**: Your Render app **must be live and accessible** at the time of grading!

---

## Troubleshooting Guide

Here are solutions to common problems you might encounter:

### "ModuleNotFoundError: No module named '...'"

Your `requirements.txt` is missing a package. Add the missing module name to `requirements.txt` and run:

```bash
pip install -r requirements.txt
```

### "Build failed on Render"

Check the **build logs** on the Render dashboard. The most common cause is a missing package in `requirements.txt`. Read the error message carefully -- it usually tells you exactly which package is missing.

### "App crashes with MemoryError"

The free Render tier has limited RAM (512 MB). To fix this:
- Use fewer or shorter documents
- Use a smaller embedding model
- Reduce your `chunk_size` to create fewer total chunks

### "Search returns irrelevant results"

- Try **reducing your `chunk_size`** (smaller chunks = more precise matching)
- Make sure your documents actually cover the topic well and contain the keywords/concepts you are searching for
- Check that your documents are not too short (very short chunks may lack enough context for good embeddings)

### "App is very slow on first load"

This is normal! The first load downloads the embedding model (~90 MB). After that, it is cached. On the free Render tier, the app also "sleeps" after 15 minutes of inactivity, so the first visit after sleep takes 30-60 seconds.

### "Git push fails"

- Make sure you **created the repository on GitHub first** (it must already exist)
- Check that the remote URL is correct: `git remote -v`
- If you get an authentication error, make sure you are logged into GitHub in your terminal (use `gh auth login` or a personal access token)

### "Streamlit run says 'command not found'"

Make sure you installed the dependencies first:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, make sure it is activated.

---

## Resources

These links will help you if you get stuck or want to learn more:

- **Streamlit documentation**: [https://docs.streamlit.io](https://docs.streamlit.io)
- **LangChain text splitters**: [https://python.langchain.com/docs/how_to/#text-splitters](https://python.langchain.com/docs/how_to/#text-splitters)
- **ChromaDB documentation**: [https://docs.trychroma.com](https://docs.trychroma.com)
- **Render deployment guide**: [https://docs.render.com/deploy-streamlit](https://docs.render.com/deploy-streamlit)
- **Git basics**: [https://docs.github.com/en/get-started/using-git](https://docs.github.com/en/get-started/using-git)
- **Sentence Transformers models**: [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)

---

Good luck, and have fun building your knowledge base!
