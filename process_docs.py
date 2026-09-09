from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the unstructured PDF data
print("Loading document...")
loader = PyPDFLoader("rulebook.pdf")
pages = loader.load()

# 2. Define the chunking strategy
# 1000 characters per chunk, 200 character overlap to maintain context across boundaries
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# 3. Execute the split
chunks = text_splitter.split_documents(pages)

print(f"Processed {len(pages)} pages into {len(chunks)} chunks.")
print("\n--- Sample Chunk ---")
print(chunks[0].page_content)

# Metadata automatically tracks which page the chunk came from
print(chunks[0].metadata)