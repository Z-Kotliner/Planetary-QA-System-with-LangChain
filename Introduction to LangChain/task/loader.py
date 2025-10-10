from langchain_community.document_loaders import DirectoryLoader, TextLoader


# Function that loads planetary data from "planets/" directory
def load_documents():
    loader = DirectoryLoader("planets", glob="*.txt", loader_cls=TextLoader)
    return loader.load()
