import os
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # Import the LangChain Groq integration

# --- Configuration ---
WORD_FILE = "my_document.docx"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Good balance
GROQ_NAME = "llama3-70b-8192"  # Or another Groq-supported model

def extract_text_from_docx(docx_path):
    """
    Extracts all text from a .docx file.
    """
    try:
        doc = Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def main():
    # 1. Extract Text from Word Document
    if not os.path.exists(WORD_FILE):
        print(f"Error: Word document not found at {WORD_FILE}")
        print("Please create a 'my_document.docx' file with some content.")
        return

    print(f"Extracting text from {WORD_FILE}...")
    document_content = extract_text_from_docx(WORD_FILE)
    if not document_content:
        return

    print("Text extracted successfully. Length:", len(document_content))

    # 2. Chunk the Text
    print("Chunking the text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust as needed
        chunk_overlap=50,  # Adjust as needed
        length_function=len,
    )
    texts = text_splitter.create_documents([document_content])
    print(f"Split into {len(texts)} chunks.")

    # 3. Generate Embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL_ID}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)
    print("Embedding model loaded.")

    # 4. Store in ChromaDB
    print(f"Storing embeddings in ChromaDB at {CHROMA_DIR}...")
    # This will create or load the ChromaDB instance
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    #vectorstore.persist()  # Persist the database to disk
    print("Embeddings stored in ChromaDB.")

    # 5. Initialize LLM (Groq)
    print(f"Initializing Groq LLM with model: {GROQ_NAME}...")
    try:
        # IMPORTANT: Set your Groq API key in the environment variable GROQ_API_KEY
        groq_llm = ChatGroq(model_name=GROQ_NAME)
        # Basic test
        groq_llm.invoke("Hello")
        print("Groq LLM initialized successfully.")
    except Exception as e:
        print(
            "Error initializing Groq.  Make sure your GROQ_API_KEY environment variable is set, and that the model is available."
        )
        print(f"Error details: {e}")
        return

    # 6. Set up RetrievalQA Chain
    print("Setting up RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        chain_type="stuff",  # "stuff" concatenates all retrieved documents
        retriever=vectorstore.as_retriever(),
    )
    print("RetrievalQA chain ready.")

    # 7. User Query Loop
    print("\n--- Ask me anything about the document! (Type 'exit' to quit) ---")
    while True:
        query = input("Your query: ")
        if query.lower() == "exit":
            break

        print("Searching and generating answer...")
        try:
            response = qa_chain.invoke({"query": query})
            print("\nAnswer:", response["result"])
            print("-" * 50)
        except Exception as e:
            print(f"An error occurred during query processing: {e}")

if __name__ == "__main__":
    main()