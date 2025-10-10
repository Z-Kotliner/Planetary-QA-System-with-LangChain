import dotenv
from qa_chain import create_qa_chain

# Load API keys as environment variables
dotenv.load_dotenv()


def main():
    # Get user query
    query = input().lower()

    # Create qa operations chains
    qa_chain = create_qa_chain()

    # Invoke chain
    chain_result = qa_chain.invoke({"input": query})

    # Print the response
    print(chain_result)

    # Print the chain info
    print(qa_chain)


if __name__ == "__main__":
    main()
