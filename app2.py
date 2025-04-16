import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Configure page settings
st.set_page_config(
    page_title="Crawler",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'crawled' not in st.session_state:
    st.session_state.crawled = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Prompt template for the LLM
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Do not try to make up an answer. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# Initialize components
@st.cache_resource
def initialize_components():
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = InMemoryVectorStore(embeddings)
    model = OllamaLLM(model="llama3")
    return embeddings, vector_store, model

embeddings, vector_store, model = initialize_components()

# Functions for crawling and processing
def load_page(url):
    try:
        loader = SeleniumURLLoader(urls=[url])
        documents = loader.load()
        return documents, None
    except Exception as e:
        return None, str(e)

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    data = text_splitter.split_documents(documents)
    return data

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Sidebar with application information
with st.sidebar:
    st.title("🕸️ Crawler")
    st.markdown("""
    ### About
    This application crawls websites and allows you to ask questions about their content using LLM technology.
    
    ### How it works:
    1. Enter a URL to crawl
    2. Wait for crawling to complete
    3. Ask questions about the content
    
    ### Technologies
    - Streamlit
    - LangChain
    - Ollama (llama3)
    """)
    
    # Optional configuration expandable section
    with st.expander("Advanced Settings"):
        st.slider("Max Context Documents", min_value=1, max_value=10, value=4)

# Main application layout
st.title("🕸️ Crawler")
st.subheader("Scrape websites and ask questions about their content")

# URL input with validation
url = st.text_input("Enter a URL to crawl:", placeholder="https://example.com")

# URL crawling section
col1, col2 = st.columns([3, 1])
with col1:
    if url:
        if not url.startswith(('http://', 'https://')):
            st.warning("Please enter a valid URL starting with http:// or https://")
with col2:
    if url and url.startswith(('http://', 'https://')):
        if st.button("Start Crawling", type="primary", use_container_width=True):
            with st.spinner("Crawling website. This may take a moment..."):
                documents, error = load_page(url)
                
                if error:
                    st.error(f"Error crawling the website: {error}")
                else:
                    with st.status("Processing content...", expanded=True) as status:
                        st.write("Splitting text into chunks...")
                        chunked_documents = split_text(documents)
                        
                        st.write(f"Indexing {len(chunked_documents)} document chunks...")
                        index_docs(chunked_documents)
                        
                        status.update(label="✅ Website crawled successfully!", state="complete")
                        st.session_state.crawled = True
                        st.balloons()

# Divider between crawling and chatting sections
st.divider()

# Chat interface
st.subheader("Ask questions about the content")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept new questions only if site has been crawled
if not st.session_state.crawled:
    st.info("Scrape a website first to start asking questions about it")
else:
    question = st.chat_input("Ask a question about the website content...")
    if question:
        # Add user question to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user question
        with st.chat_message("user"):
            st.write(question)
        
        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                retrieved_documents = retrieve_docs(question)
                context = "\n\n".join([doc.page_content for doc in retrieved_documents])
                
                # Get answer from LLM
                answer = answer_question(question, context)
                
                # Display the answer
                st.write(answer)
                
                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Show sources in expandable section
                with st.expander("View sources"):
                    for i, doc in enumerate(retrieved_documents):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:300] + "...")

# Footer
st.markdown("---")
st.caption("Crawler - Built with Streamlit and LangChain")