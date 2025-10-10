from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os


# A function to initialize HF embeddings
def get_embeddings():
    embeddings_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    )
    return embeddings_model
