import streamlit as st
from rag_system import initialize_collection, add_document, query_documents

st.title("Multilingual Chatbot with RAG System")

collection = initialize_collection()

# Document upload
uploaded_file = st.file_uploader("Upload Document (PDF/TXT)", type=['txt', 'pdf'])
if uploaded_file:
    content = None
    try:
        if uploaded_file.type == 'application/pdf':
            import PyPDF2, io
            pdf = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            content = " ".join(page.extract_text() for page in pdf.pages)
        else:
            content = uploaded_file.read().decode('utf-8', errors='ignore')
        
        add_document(collection, content)
        st.success("Document added Successfully!")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Query section
target_lang = st.selectbox("Output Language", ['en', 'es', 'fr', 'de', 'zh'])
if query := st.text_input("Your Question"):
    for idx, result in enumerate(query_documents(collection, query, target_lang), 1):
        st.write(f"{idx}. {result}")
