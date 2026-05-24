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

# Initialize API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Streamlit App Configuration
st.set_page_config(page_title="PDF Insight Assistant", layout="wide")

st.title("📄 PDF Insight Assistant")
st.markdown(
    "#### Upload PDF documents and ask questions using Gemini-powered Retrieval Augmented Generation (RAG)."
)

# Initialize Gemini LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.1,
        max_tokens=2048
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {str(e)}")
    st.stop()

# Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert AI assistant. Use the provided context to answer the user's question accurately and clearly.
    
    If the answer is not present in the context, say:
    "The answer could not be found in the uploaded documents."

    <context>
    {context}
    </context>

    Question: {input}
    """
)

DATA_FOLDER = "./data"

# --- Utility Functions ---

def clean_previous_data():
    """Clean the data folder on app reload."""
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.makedirs(DATA_FOLDER)

def save_uploaded_file(uploaded_file):
    """Save uploaded PDF file."""
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    return file_path

def initialize_vector_store_from_upload(uploaded_files):
    """Create embeddings and vector store from uploaded PDFs."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004"
    )

    documents = []

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)

        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_documents = text_splitter.split_documents(documents)

    # Create FAISS vector database
    vectors = FAISS.from_documents(final_documents, embeddings)

    return vectors

def create_retrieval_chain_with_context(llm, vectors):
    """Create retrieval chain."""
    
    retriever = vectors.as_retriever()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt_template
    )

    return create_retrieval_chain(retriever, document_chain)

def handle_user_question(question, retrieval_chain):
    """Generate answer from uploaded documents."""

    responses = {}

    start = time.process_time()

    try:
        response_with_context = retrieval_chain.invoke({
            'input': question
        })

        responses['with_context'] = {
            "response_time": time.process_time() - start,
            "answer": response_with_context.get(
                'answer',
                "No answer generated."
            ),
            "context": response_with_context.get(
                'context',
                []
            )
        }

    except Exception as e:
        st.error(f"Error during response generation: {e}")

        responses['with_context'] = {
            "response_time": None,
            "answer": "Error generating response.",
            "context": []
        }

    return responses

# --- Streamlit Layout ---

# Clean old data on refresh
clean_previous_data()

with st.sidebar:
    st.header("Assistant Panel")

    st.markdown("""
    Upload PDF documents and ask questions about their content.

    - **Model:** Gemini 2.5 Pro
    - **Embedding:** Gemini Text Embedding
    - **Vector Store:** FAISS
    """)

# Upload Section
st.subheader("Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    if "vectors" not in st.session_state:
        st.session_state.vectors = None

    if st.button("Initialize Document Embedding", key="init_embedding"):

        with st.spinner("Processing documents and creating vector store..."):

            try:
                st.session_state.vectors = initialize_vector_store_from_upload(
                    uploaded_files
                )

                st.success("Vector store initialized successfully.")

            except Exception as e:
                st.error(f"Initialization failed: {e}")
                st.session_state.vectors = None

# Question Input
question = st.text_input(
    "Ask a question about the uploaded PDFs:",
    help="Ask anything related to the uploaded documents."
)

# Generate Response
if question and st.session_state.get("vectors") is not None:

    st.session_state.retrieval_chain = (
        create_retrieval_chain_with_context(
            llm,
            st.session_state.vectors
        )
    )

    with st.spinner("Generating answer..."):

        responses = handle_user_question(
            question,
            st.session_state.retrieval_chain
        )

    if 'with_context' in responses:

        response_data = responses['with_context']

        st.subheader("Generated Answer")

        st.text_area(
            "Answer",
            response_data['answer'],
            height=300
        )

        if response_data['response_time'] is not None:
            st.info(
                f"Response Time: {response_data['response_time']:.2f} seconds"
            )

        with st.expander("View Retrieved Context"):

            if response_data['context']:

                for i, doc in enumerate(response_data['context']):

                    st.markdown(
                        f"**Chunk {i+1} "
                        f"(Page: {doc.metadata.get('page', 'N/A')})**"
                    )

                    st.text(doc.page_content)

                    st.markdown("---")

            else:
                st.write("No relevant chunks retrieved.")

elif question and st.session_state.get("vectors") is None:

    st.warning(
        "Please upload PDF documents and initialize embeddings first."
    )

# Footer
st.markdown("""
---
*This application uses Gemini LLMs and Retrieval-Augmented Generation (RAG)
to answer questions based on uploaded PDF documents.*
""")
