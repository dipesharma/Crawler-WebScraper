import streamlit as st 
import tempfile
import os
from langchain_community.document_loaders import SeleniumURLLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks, and your primary goal is to provide helpful and concise answers based on the retrieved context. Below are some instructions to guide your responses:

1. **General Response**: The default answer should be 3-4 sentences. However, if the user asks for more detail, respond in a longer, more descriptive manner. If the user requests a shorter answer, be brief and to the point.

2. **Response Style**: Respond with a friendly, conversational tone, while maintaining professionalism. Be polite and engaging. If the conversation is formal, keep the tone formal. If it's casual, be informal and warm.

3. **Handling Uncertainty**: If the answer to a question is not clear or you don't know it, say: "I'm sorry, I don't know the answer, but I can try to help with related topics." Avoid making up answers.

4. **Handling Clarifications**: If the question is vague, ask the user for clarification in a polite manner. For example: "Could you clarify what you're asking?" or "Could you provide more context?"

5. **Context Focus**: Use the retrieved context and make sure the answer is **directly related** to the provided information. Do not provide answers based on external knowledge unless explicitly stated.

6. **Polite Conversation**: Respond appropriately to casual interactions like greetings or thanks. For example, if the user says "Thank you," respond with "You're welcome!" or something polite like "No problem, happy to help!"

7. **Engagement**: If the conversation feels one-sided (e.g., the user asks several questions in a row), ask follow-up questions or offer to help with more information. For example: "Would you like more details?" or "Can I assist you with anything else?"

8. **Fallback**: If you don't have any context to answer a question, inform the user politely and offer to help with related topics.

9. **Your Instructions**: Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Do not try to make up an answer. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer: 
"""



# Embedding and model setup
embeddings = OllamaEmbeddings(model="llama3")
model = OllamaLLM(model="llama3")

# Initialize memory store for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores (question, answer) pairs
    
    
# Initialize vector store in session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embeddings)

# Functions
def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    st.session_state.vector_store.add_documents(documents)

def retrieve_docs(query):
    return st.session_state.vector_store.similarity_search(query)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
st.title("🕷️ Crawler")

# Input section
url = st.text_input("Enter URL to scrape:")
uploaded_file = st.file_uploader("Or upload a PDF file", type=["pdf"])

# Load documents from either source
documents = None

if url and st.button("Crawl URL"):
    with st.spinner("Scraping and processing content..."):
        documents = load_page(url)
        chunked_documents = split_text(documents)
        index_docs(chunked_documents)
        st.session_state.chat_history = []  # Clear memory
        st.success("Scraped and indexed the page content!")

elif uploaded_file and st.button("Process PDF"):
    with st.spinner("Reading and processing PDF..."):
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Write content to temp file
            temp_file_path = temp_file.name  # Get the path of the temp file

        loader = PyMuPDFLoader(temp_file_path)  # Load the PDF from temp file
        documents = loader.load()
        chunked_documents = split_text(documents)
        index_docs(chunked_documents)
        st.session_state.chat_history = []  # Clear memory
        st.success("PDF processed and indexed!")

        # Optionally, delete the temp file after processing
        os.remove(temp_file_path)
# Chat Input
question = st.chat_input("Ask a question about the content...")

if question:
    # Display previous messages (both user and assistant)
    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    # Now handle the new question
    st.chat_message("user").write(question)
    
    # Retrieve documents
    retrieved_documents = retrieve_docs(question)

    # Combine past Q&A into context (up to the last 3 interactions)
    chat_history_context = ""
    for q, a in st.session_state.chat_history[-3:]:  # Limit to last 3 exchanges
        chat_history_context += f"Q: {q}\nA: {a}\n"

    # Add newly retrieved content from documents
    docs_context = "\n\n".join([doc.page_content for doc in retrieved_documents])

    # Final context includes both history + fresh info
    context = chat_history_context + "\n" + docs_context

    # Get the model's answer
    answer = answer_question(question, context)
    st.chat_message("assistant").write(answer)
    
    # Append current Q&A to chat history
    st.session_state.chat_history.append((question, answer))
