from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from retriever import similarity_search


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


# Global Tool list to be used with the llm and run_tools function
available_tools = [get_planet_distance, get_planet_revolution, get_planet_info]


# A function that implements the logic for running tools
def run_tools(ai_msg: AIMessage):
    available_tools_map = {tool.name: tool for tool in available_tools}

    # Identify the first tool to run
    tool_to_call = ai_msg.tool_calls[0]

    # Extract the tool name and arguments
    tool_name = tool_to_call['name']
    tool_args = tool_to_call['args']

    # Get the tool to run from the available tools
    tool_to_run = available_tools_map[tool_name]

    # Execute identified tool with the query
    response = tool_to_run.invoke(tool_args)

    return response
