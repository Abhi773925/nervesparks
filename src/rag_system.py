import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from utils import detect_language, translate_text

# Initialize ChromaDB with specific settings
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from utils import detect_language, translate_text

# Get the absolute path to the chroma_db directory
persist_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

# Configure ChromaDB settings
settings = Settings(
    persist_directory=persist_directory,
    is_persistent=True,
    anonymized_telemetry=False
)

# Initialize the persistent client
client = chromadb.PersistentClient(path=persist_directory, settings=settings)
# If the above doesn't work, try this alternative:
# client = chromadb.EphemeralClient()

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def initialize_collection():
    try:
        return client.get_collection(name="multilingual_docs", embedding_function=embedder)
    except:
        return client.create_collection(name="multilingual_docs", embedding_function=embedder)

def add_document(collection, text):
    # Split text into smaller chunks for better retrieval
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.strip()],
            metadatas=[{"language": detect_language(chunk)}],
            ids=[f"doc_{len(collection.get()['ids']) + 1}_{i}"]
        )

def query_documents(collection, query_text, target_lang='en', n_results=1):
    # Get most relevant chunk
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    if not results['documents'][0]:
        return ["No relevant information found."]
        
    best_match = results['documents'][0][0]
    source_lang = detect_language(best_match)
    
    if source_lang != target_lang:
        return [translate_text(best_match, source_lang, target_lang)]
    return [best_match]
