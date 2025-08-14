import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from utils import detect_language, translate_text

# Initialize ChromaDB with specific settings
client = chromadb.HttpClient(host="localhost", port=8000)  # For local HTTP server
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
