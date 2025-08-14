# Multilingual RAG System

A simple multilingual Retrieval-Augmented Generation (RAG) system that can process documents in multiple languages and answer queries in the user's preferred language.

## Structure

```
src/
├── app.py          # Streamlit interface
├── rag_system.py   # Core RAG functionality
└── utils.py        # Language detection & translation
```

## Flow

1. **Document Processing**
   - Upload PDF/TXT files
   - Text is split into manageable chunks
   - Each chunk is stored with language metadata

2. **Query Processing**
   - User inputs a question in any language
   - System finds the most relevant document chunk
   - Response is translated to user's preferred language

3. **Components**
   - ChromaDB for vector storage and retrieval
   - MarianMT for translations
   - langdetect for language detection

## Requirements

```
chromadb
streamlit
transformers
torch
PyPDF2
langdetect
```

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run src/app.py
   ```

3. Use the interface to:
   - Upload documents
   - Select your preferred language
   - Ask questions about your documents
