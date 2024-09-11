import os
import streamlit as st
from model import ChatModel
import rag_util

# Directory for file storage
FILES_DIR = os.path.join(os.getcwd(), "files")
os.makedirs(FILES_DIR, exist_ok=True)

st.title("Chatbot LLM & RAG Assistant")

@st.cache_resource
def load_model():
    model = ChatModel(model_id="google/gemma-2-2b-it", device="cuda")
    return model

@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder

model = load_model()  # Load model and save into cache
encoder = load_encoder()

def save_file(uploaded_file):
    """Helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Sidebar
with st.sidebar:
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
    k = st.number_input("k", 1, 10, 3)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        file_paths = [save_file(uploaded_file) for uploaded_file in uploaded_files]
        docs = rag_util.load_and_split_pdfs(file_paths)
        DB = rag_util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)
    else:
        DB = None  # Ensure DB is None when no files are uploaded

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and generate responses
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if DB:
            # Use RAG approach if PDFs are uploaded
            retrieved_docs = DB.similarity_search(prompt, k=k)
            if retrieved_docs:
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                answer = model.generate(prompt, context=context, max_new_tokens=max_new_tokens)
            else:
                answer = "Sorry, I couldn't find relevant information."
        else:
            # If there is no pdf ,use llm for general purpose
            answer = model.generate(prompt, context=None, max_new_tokens=max_new_tokens)
        
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("""
<style>
.streamlit-chat__message {
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)
