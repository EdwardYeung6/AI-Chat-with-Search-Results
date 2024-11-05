import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import List
import openai
from index import css, bot_template, user_template
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI with custom base path
openai.api_base = os.getenv("BASE_API_PATH")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.default_headers = {"x-foo": "true"}

logger.debug(f"OpenAI API Base: {openai.api_base}")
logger.debug(f"OpenAI API Key: {'Set' if openai.api_key else 'Not Set'}")

def init_search():
    """Initialize Google Search"""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        logger.debug(f"Google API Key: {'Set' if google_api_key else 'Not Set'}")
        logger.debug(f"Google CSE ID: {'Set' if google_cse_id else 'Not Set'}")
        
        return GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
    except Exception as e:
        logger.error(f"Error initializing search: {str(e)}")
        raise

def search_and_get_content(query: str, search: GoogleSearchAPIWrapper) -> List[Document]:
    """Search Google and return first 5 results as documents"""
    try:
        logger.debug(f"Searching for query: {query}")
        search_results = search.results(query, 5)
        logger.debug(f"Got {len(search_results)} search results")
        
        documents = []
        for result in search_results:
            content = result.get("snippet", "")
            link = result.get("link", "")
            if content and link:
                doc = Document(page_content=content, metadata={"source": link})
                documents.append(doc)
                logger.debug(f"Added document from {link}")
        return documents
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        st.error(f"Search error: {str(e)}")
        return []

def create_vectorstore(documents: List[Document]):
    """Create vector store from documents"""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings with custom base URL
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("BASE_API_PATH")
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    """Create conversation chain"""
    try:
        logger.debug("Initializing ChatOpenAI")
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("BASE_API_PATH")
        )
        
        logger.debug("Creating conversation memory")
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Create a prompt template for the system message
        system_template = """You are a helpful assistant that can answer questions based on the provided documents 
        and also offer relevant suggestions or information from your general knowledge when appropriate. 
        If you provide information outside of the documents, please indicate this clearly.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that."""
        
        prompt = PromptTemplate(
            template=system_template,
            input_variables=["context", "question"]
        )
        
        logger.debug("Creating conversation chain")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,        
            verbose=True,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        logger.debug("Conversation chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        raise

def handle_userinput(user_question):
    """Handle user input"""
    try:
        logger.debug(f"Processing user question: {user_question}")
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            logger.debug(f"Message {i} rendered")
    except Exception as e:
        logger.error(f"Error handling user input: {str(e)}")
        st.error(f"Error processing your question: {str(e)}")

def main():
    try:
        st.set_page_config(
            page_title="Chat with Search Results",
            page_icon="🔍",
            layout="wide"
        )
        
        st.write(css, unsafe_allow_html=True)
        
        logger.debug("Initializing search")
        search = init_search()
        
        st.title("Chat with Search Results 🔍")

        # Initialize session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if "documents" not in st.session_state:
            st.session_state.documents = None

        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search for a topic...", key="searchInput")
        with col2:
            if st.button("Search", key="searchButton"):
                if not search_query.strip():
                    st.warning("Please enter a search query")
                else:
                    with st.spinner('Searching...'):
                        logger.debug(f"Processing search query: {search_query}")
                        st.session_state.documents = search_and_get_content(search_query, search)
                        if st.session_state.documents:
                            vectorstore = create_vectorstore(st.session_state.documents)
                            if vectorstore is not None:
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                st.success("Search completed! You can now chat about the results.")
                            else:
                                st.error("Failed to create vector store. Please check your API configuration.")
                        else:
                            st.warning("No results found. Please try a different search query.")

        # Display search results in sidebar
        with st.sidebar:
            st.subheader("Search Results")
            if st.session_state.documents:
                for doc in st.session_state.documents:
                    with st.expander(f"Source: {doc.metadata['source']}", expanded=False):
                        st.write(doc.page_content)

        # Chat interface
        st.subheader("Chat")
        
        # Chat input
        user_question = st.text_input("Enter your question...", key="questionInput")
        
        if user_question:
            if st.session_state.conversation is None:
                st.warning("Please perform a search first before chatting.")
            else:
                handle_userinput(user_question)
    
    except Exception as e:
        logger.error(f"Main application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()