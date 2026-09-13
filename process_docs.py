from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# 1. Load and Split (Your existing code)
print("Loading and chunking document...")
loader = PyPDFLoader("rulebook.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

# 2. Define the Embedding Model
print("Downloading embedding model (this may take a minute on the first run)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create and Store in ChromaDB
print("Converting text to vectors and saving to database...")
persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Success! Database saved locally in the '{persist_directory}' folder.")

# 4. Test the Similarity Search
query = "What is the team size limit?" # Change this to a real question about your PDF
results = vectorstore.similarity_search(query, k=2) # k=2 returns the top 2 closest chunks

print("\n--- Top Search Result ---")
print(results[0].page_content)