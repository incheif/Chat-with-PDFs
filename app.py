import streamlit as st
import os
import shutil
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# --- Initialization and Configuration ---

# Initialize API key from Streamlit secrets.
# This assumes st.secrets["GOOGLE_API_KEY"] is set in your Streamlit environment.
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Streamlit app title
st.set_page_config(page_title="Research Paper Analysis (Gemini RAG)", layout="wide")
st.title("Research Paper Analysis using Retrieval Augmented Generation (Gemini)")
st.markdown("#### Leverage the power of Google's Gemini models to analyze research papers.")

# Initialize the Gemini LLM for conversational responses
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", # Changed model from gemini-2.5-flash to gemini-2.5-pro
        temperature=0.1,
        max_tokens=2048
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {str(e)}")
    st.stop()

# Define prompt template for RAG
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert academic assistant. Use the following context to answer the user's question accurately and thoroughly.
    If the answer is not found in the context, state that clearly.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

DATA_FOLDER = "./data"

# --- Utility Functions ---

def clean_previous_data():
    """Clean the `data` folder on app reload or restart."""
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.makedirs(DATA_FOLDER)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the data folder and return the file path."""
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def initialize_vector_store_from_upload(uploaded_files):
    """Initialize vector embeddings and FAISS vector store from uploaded files using Google Generative AI Embeddings."""
    # Use the GoogleGenerativeAIEmbeddings for consistency
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = []

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    
    # Create the FAISS vector store
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

def create_retrieval_chain_with_context(llm, vectors):
    """Create a retrieval chain using the Gemini LLM and vector store."""
    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)

def handle_user_question(question, retrieval_chain):
    """Handle user question and generate responses with context."""
    responses = {}

    start = time.process_time()
    try:
        # Invoke the RAG chain
        response_with_context = retrieval_chain.invoke({'input': question})
        responses['with_context'] = {
            "response_time": time.process_time() - start,
            "answer": response_with_context.get('answer', "No answer generated."),
            "context": response_with_context.get('context', [])
        }
    except Exception as e:
        st.error(f"Error during context-based response: {e}")
        responses['with_context'] = {"response_time": None, "answer": "Error generating response.", "context": []}

    return responses

# --- Streamlit Application Layout ---

# Clean data folder on reload
clean_previous_data()

with st.sidebar:
    st.header("Assistant Panel")
    st.markdown("""
        Upload your PDF research papers and ask specific questions about their content.
        
        - **Model:** Gemini 2.5 Pro (Updated)
        - **Embedding:** Google Generative AI Embedding 001
    """)

# 1. Upload Section
st.subheader("Upload Research PDFs")

uploaded_files = st.file_uploader(
    "Choose one or more Research documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    # Use session state to manage the initialization status
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    
    if st.button("Initialize Document Embedding", key="init_embedding"):
        with st.spinner("Processing uploaded documents and creating vector store..."):
            try:
                st.session_state.vectors = initialize_vector_store_from_upload(uploaded_files)
                st.success("Vector store initialized successfully.")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
                st.session_state.vectors = None


# 2. Question Input Section
question = st.text_input(
    "Enter your question :",
    help="Ask a specific question related to the uploaded documents after initialization."
)

# 3. Process the question if a vector store is ready
if question and st.session_state.get("vectors") is not None:
    st.session_state.retrieval_chain = create_retrieval_chain_with_context(llm, st.session_state.vectors)
    
    with st.spinner("Generating answer..."):
        responses = handle_user_question(question, st.session_state.retrieval_chain)

    if 'with_context' in responses:
        response_data = responses['with_context']
        st.subheader("Generated Answer")
        st.text_area("Answer", response_data['answer'], height=300)

        st.info(f"Response Time: {response_data['response_time']:.2f} seconds")

        with st.expander("Source Context (Relevant Document Chunks):"):
            if response_data['context']:
                for i, doc in enumerate(response_data['context']):
                    st.markdown(f"**Chunk {i+1} (Source Page: {doc.metadata.get('page', 'N/A')})**")
                    st.text(doc.page_content)
                    st.markdown("---")
            else:
                st.write("No relevant document chunks were retrieved for the question.")

elif question and st.session_state.get("vectors") is None:
    st.warning("Please upload documents and click 'Initialize Document Embedding' first.")

# 4. Footer or Additional Information Section
st.markdown("""
    ---
    *Disclaimer: This tool uses LLMs and RAG techniques for document analysis. Always verify important scientific or factual conclusions with the original source papers.*
""")