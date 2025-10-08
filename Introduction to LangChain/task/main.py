import os

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys as environment variables
dotenv.load_dotenv()

# Load planetary data from "planets/" directory
loader = DirectoryLoader("planets", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Create a recursive Character text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

# Split documents into chunks
chunked_documents = text_splitter.split_documents(documents)

# Initialize HF embeddings
embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_API_KEY"),
)

# Convert document data into vector embedding and store in Chroma vector store
db = Chroma.from_documents(documents, embeddings_model)

# Fetch relevant documents based on user queries using similarity search
query = input().lower()
docs = db.similarity_search(query)

# Print the contents of the most similar document.
print(docs[0].page_content)
