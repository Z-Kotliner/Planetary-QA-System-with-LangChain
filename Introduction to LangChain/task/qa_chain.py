from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools import available_tools, run_tools

# A function that creates a chain of operations of prompting -> llm -> tool_runner
def create_qa_chain():
    # Instantiate the llm with model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6,
        max_retries=2)

    # Create messages
    messages = [
        ("system",
         "You are a helpful AI assistant that provide answers to user question. Use the tools provided to answer."),
        ("user", "{input}")
    ]

    # Bind the llm with the available_tools
    llm_with_tools = llm.bind_tools(available_tools)

    # Create a prompt
    prompt = ChatPromptTemplate.from_messages(messages)

    # Create a chain composing prompt, llm bound with tools and run_tools function
    chain = prompt | llm_with_tools | run_tools

    return chain
