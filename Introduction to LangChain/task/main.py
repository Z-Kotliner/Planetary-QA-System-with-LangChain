import os

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys as environment variables
dotenv.load_dotenv()


# Define the PlanetDistanceSun tool.
@tool("PlanetDistanceSun")
def get_planet_distance(planet_name: str) -> str:
    """
    Calculates the distance of a planet from the Sun in the form of a sentence.

    :param planet_name: The name of a Planet.
    :type planet_name: str

    :return: The distance in the form of a sentence answer.
    :rtype: str

    """
    match planet_name:
        case "Earth":
            return "Earth is approximately 1 AU from the Sun."
        case "Mars":
            return "Mars is approximately 1.5 AU from the Sun."
        case "Jupiter":
            return "Jupiter is approximately 5.2 AU from the Sun."
        case "Pluto":
            return "Pluto is approximately 39.5 AU from the Sun."
        case _:
            return f"Information about the distance of {planet_name} from the Sun is not available in this tool."


# Define the PlanetRevolutionPeriod tool
@tool("PlanetRevolutionPeriod")
def get_planet_revolution(planet_name: str) -> str:
    """
    Provides the approximate revolution period around the Sun in Earth years in the form of a sentence.

    :param planet_name: The name of a Planet.
    :type planet_name: str

    :return: The approximate revolution period around the Sun in Earth years.
    :rtype: str

    """
    match planet_name:
        case "Earth":
            return "Earth takes approximately 1 Earth year to revolve around the Sun."
        case "Mars":
            return "Mars takes approximately 1.88 Earth years to revolve around the Sun."
        case "Jupiter":
            return "Jupiter takes approximately 11.86 Earth years to revolve around the Sun."
        case "Pluto":
            return "Pluto takes approximately 248 Earth years to revolve around the Sun."
        case _:
            return f"Information about the revolution period of {planet_name} is not available in this tool."


# Define the PlanetGeneralInfo tool.
@tool("PlanetGeneralInfo")
def get_planet_info(planet_name: str) -> str:
    """
    Provides information about a planet using similarity search over documents in planets/ directory.

   :param planet_name: The name of a Planet.
   :type planet_name: str

   :return: General information about a planet using similarity search over documents.
   :rtype: str

   """
    search_result = similarity_search(planet_name)
    if not search_result:
        return f"Additional information for {planet_name} is not available in this tool."
    else:
        return search_result


# Define a function that performs similarity search on planets/ directory using embeddings
def similarity_search(planet_name: str) -> str:
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

    # Fetch relevant documents based on passed argument(user query) using similarity search
    query = planet_name
    docs = db.similarity_search(query)

    # Return the contents of the most similar document.
    return docs[0].page_content


def execute_search(user_input: str):
    # Tool list to be used with the llm
    available_tools = [get_planet_distance, get_planet_revolution, get_planet_info]
    available_tools_map = {tool.name: tool for tool in available_tools}

    # Instantiate the llm with model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_retries=2)

    # Bind the llm with the available_tools
    llm_with_tools = llm.bind_tools(available_tools)

    # Retrieve the ai_message containing tool_calls
    ai_msg = llm_with_tools.invoke(user_input)

    # Extract the .tool_calls list
    tool_calls = ai_msg.tool_calls

    # Identify the first tool to run
    tool_to_call = ai_msg.tool_calls[0]

    # Extract the tool name and arguments
    tool_name = tool_to_call['name']
    tool_args = tool_to_call['args']

    # Get the tool to run from the available tools
    tool_to_run = available_tools_map[tool_name]

    # Execute identified tool with the query
    response = tool_to_run.invoke(tool_args)

    # Print the response
    print(response)

    # Print the tool_calls
    print(tool_calls)


def main():
    # Get user query
    query = input().lower()

    # Execute search with query
    execute_search(query)


if __name__ == "__main__":
    main()
