# 🚀 Advanced RAG System By TURJOY

## 🧠 Overview

**Advanced RAG System By TURJOY** is a powerful **Retrieval-Augmented Generation (RAG)** system built to process and query **large PDF documents (500+ pages)** efficiently and accurately.  
It integrates **OpenAI’s `gpt-4o-mini`** for intelligent response generation and **`text-embedding-3-small`** for high-quality embeddings.  

This system combines **semantic chunking**, **hybrid retrieval**, and **MMR reranking** to provide high-quality, context-aware answers with full **source traceability (page-level citations)**.  
A clean and interactive **Streamlit UI** allows easy document upload, indexing, querying, and monitoring.

---

## ✨ Features

✅ **Large Document Processing**  
Handles PDFs with **500+ pages** efficiently.

✅ **Semantic Chunking**  
Intelligent token-based splitting for context-aware chunking.

✅ **OpenAI Integration**  
Uses:
- `gpt-4o-mini` → generating answers & follow-ups  
- `text-embedding-3-small` → generating embeddings for document chunks and queries  

✅ **ChromaDB Vector Store (Persistent)**  
Stores embeddings and chunks for fast similarity search.

✅ **Hybrid Retrieval**  
Combines **semantic vector search + keyword search** to improve recall.

✅ **Maximal Marginal Relevance (MMR) Reranking**  
Improves diversity and reduces redundancy among retrieved chunks.

✅ **Source Traceability (Citations)**  
Answers include **PDF page-number references** for transparency.

✅ **Streamlit Web UI**  
Interactive interface for:
- PDF upload  
- Document indexing  
- Querying  
- Monitoring system info  

✅ **Robust Error Handling**  
Helpful troubleshooting and detailed errors inside the UI.

---

## 🛠️ Technology Stack

| Component | Tools |
|----------|------|
| Language | Python |
| UI | Streamlit |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Database | ChromaDB |
| PDF Extraction | PyPDF |
| Tokenizer | Tiktoken |
| Similarity + MMR | NumPy, Scikit-learn |
| Config Management | Python-dotenv |

---

## ⚙️ Setup & Installation

### ✅ Prerequisites

- Python **3.8+**
- OpenAI API Key (Get it from: https://platform.openai.com/api-keys)

---

### 🔧 Installation Steps

#### 1️⃣ Clone the repository (or place files in a folder)

```bash
# git clone <repo_url>
# cd advanced-rag-system
````

#### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Set up OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
```

✅ You can also enter the API key directly from the Streamlit sidebar during runtime.

---

## 🚀 Run the Application

Activate virtual environment:

```bash
source venv/bin/activate
```

Start Streamlit app:

```bash
streamlit run main.py
```

Open in browser:

📍 `http://localhost:8501`

---

## 🖥️ Usage Guide

### ✅ Step 1: Initialize System

* Enter OpenAI API Key in the sidebar (if not using `.env`)
* Click **🚀 Initialize System**
* Wait until the system shows **System Ready**

---

### ✅ Step 2: Upload & Process PDF

Go to **Document Processing Tab**:

1. Upload a PDF
2. Click **🔄 Process Document**
3. Wait for indexing completion
4. Processed files will appear in **Indexed Documents Table**

---

### ✅ Step 3: Ask Questions

Go to **Query Documents Tab**:

1. Enter your question
2. Adjust **📊 Sources slider**
3. Click **🔍 Search**
4. Output includes:

   * AI Answer
   * Source chunks
   * Page citations

---

### ✅ Step 4: Monitor System Stats

Go to **System Info Tab**:

* Model and stack info
* Indexed doc count
* Page/chunk stats
* Troubleshooting tips

---

## ⚠️ Troubleshooting

Common issues and solutions are available inside the app under **System Info**:

* Missing OpenAI key
* Network/API errors
* PDF processing issues
* ChromaDB errors
* No results found
* Memory overload for large PDFs

---

## 🤝 Contributing

Contributions are welcome!
If you have suggestions, bug fixes, or feature ideas, feel free to open an issue or submit a pull request.

---

## 📌 Author

**TURJOY**
🚀 Advanced RAG System By TURJOY


<img width="1832" height="827" alt="image" src="https://github.com/user-attachments/assets/e546db09-9afe-4ca8-8c48-e891785297af" />

