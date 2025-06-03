import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import glob

load_dotenv()

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def load_documents(directory_path):
    """Load documents from the specified directory"""
    docs = []
    pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)
    
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            file_docs = loader.load()
            docs.extend(file_docs)
        except Exception as e:
            st.warning(f"Error loading file {os.path.basename(pdf_file)}: {str(e)}")
    
    return docs

def create_vectorstore(docs):
    """Create vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings()
    )
    return vectorstore

def format_docs(docs):
    """Format documents for display"""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore):
    """Create RAG chain"""
    retriever = vectorstore.as_retriever()
    
    prompt_template = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Based on the given context, answer the question in detail. "
        "If you cannot find the answer in the context, say so.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    st.title("📚 Document Q&A System")
    
  
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            
            temp_dir = "temp_docs"
            os.makedirs(temp_dir, exist_ok=True)
            
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getvalue())
            
          
            with st.spinner("Processing documents..."):
                docs = load_documents(temp_dir)
                st.session_state.vectorstore = create_vectorstore(docs)
                st.success("Documents processed successfully!")
    
    
    if st.session_state.vectorstore:
        
        question = st.text_input("Ask a question about your documents:")
        
        if question:
            with st.spinner("Generating answer..."):
                rag_chain = create_rag_chain(st.session_state.vectorstore)
                answer = rag_chain.invoke(question)
                
                st.markdown("### Answer")
                st.write(answer)
    else:
        st.info("Please upload some PDF documents to get started!")

if __name__ == "__main__":
    main() 