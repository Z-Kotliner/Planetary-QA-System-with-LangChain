from vector_store import create_chroma_store


# A function that performs similarity search on planets/ directory using embeddings
def similarity_search(planet_name: str) -> str:
    chroma_db = create_chroma_store()

    # Fetch relevant documents based on passed planer_name using similarity search
    docs = chroma_db.similarity_search(planet_name)

    # Return the contents of the most similar document.
    return docs[0].page_content
