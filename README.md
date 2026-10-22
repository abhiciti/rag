---
title: PDF RAG Knowledge Base
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# PDF Knowledge Base RAG Assistant AST1 and AST 2
An interactive local RAG application processing `AST-1.pdf` and `AST-2.pdf` utilizing the open-weights `Qwen2.5-1.5B-Instruct` model and `BAAI/bge-small-en-v1.5` embeddings.

### How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Place `AST-1.pdf` and `AST-2.pdf` in the root directory.
3. Run the Gradio App: `python app.py`
