import loader
import embedder
from langchain_chroma import Chroma


# A function that creates a chroma vector store from the documents in the planets/ directory
def create_chroma_store():
    # Load documents in planets/ directory
    documents = loader.load_documents()

    # Init embedding model
    embeddings_model = embedder.get_embeddings()

    # Convert document data into vector embedding and store in Chroma vector store
    vector_db = Chroma.from_documents(documents, embeddings_model)

    return vector_db
