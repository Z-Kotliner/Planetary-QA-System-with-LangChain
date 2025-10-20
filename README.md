# Planetary QA System with LangChain
---
## Project Overview
This project serves as a **practical introduction to LangChain**, designed to use Large Language Models (LLMs) in your applications. The rise of LLMs is changing how we process data and develop software, and LangChain provides the essential tools to seamlessly integrate these models into your workflow.

It is a **QA system** that answers questions about planets by leveraging LangChain. The system integrates LangChain's LLM chain of operations, embeddings, document loading, retrievers, and a vector database (Chroma) to create an intelligent, LLM-driven application.

---

## Key Concepts Covered:
- **LangChain LLM Chain**: Use of LangChain's chain of operations to design intelligent workflows.
- **Tooling and Embeddings**: Integration of tools and embeddings (e.g., `HuggingFaceEndpointEmbeddings`) for enhancing LLM performance.
- **Loading External Documents**: How to load and manage external documents or data into your application.
- **Retrievers and Vector Databases**: Use of vector databases like Chroma for storing and retrieving relevant documents based on queries.

## Features

- **Planetary QA System**: Get answers about different planets based on available documents and LLM capabilities.
- **External Document Integration**: Load external data (e.g., planet facts) for context.
- **Retriever and Vector Database**: Use Chroma to store and retrieve relevant information for answering questions.
- **LLM-driven Workflow**: Implement a LangChain LLM chain to handle complex operations.

---

## Tools and Technologies

This project utilizes a variety of powerful tools and technologies to build an intelligent, LLM-driven application. These tools enable seamless integration of LLMs with external data sources and vector databases for efficient retrieval and enhanced performance.

- **LangChain**: A framework for building LLM-powered applications, enabling easy integration with various LLMs and tools.
  - `langchain-core`: The core package for LangChain's LLM and chain functionalities.
  - `langchain-community`: A community-driven set of additional modules and extensions for LangChain.
  - `langchain-groq`: Integration with **Groq** for powerful AI model capabilities.
  - `langchain-chroma`: Integration with Chroma, a vector database for efficient document retrieval.
  
- **Chroma**: A vector database used for storing and retrieving embeddings, enabling semantic search across large collections of documents.
  
- **HuggingFaceEndpointEmbeddings**: Provides embeddings for external documents using models hosted on Hugging Face, helping improve the accuracy of document retrieval.

- **Groq**: An external LLM provider integrated through LangChain's `langchain-groq` module. **Groq** models provide high-performance, specialized AI capabilities that can be used for the LLM-driven tasks in this project.

- **Python-dotenv**: Manages environment variables to securely store API keys and other sensitive data for the project.

---

## Installation Steps

1. **Clone the repository**:
     ```bash
    git clone https://github.com/Z-Kotliner/Planetary-QA-System-with-LangChain.git
    cd introduction-to-langchain

2. **Set up a Python virtual environment (optional but recommended):**
     ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. ** Install the required dependencies:**
     ```bash
    pip install -r requirements.txt

4. **Set up your environment variables by creating a .env file. Here's an example .env file:**
     ```bash
    HF_API_KEY=your-huggingface-api-key
    GROQ_API_KEY=your-groq-api-key


Replace the placeholders with your actual API keys.

---

This project was done as part of the fulfilment of the core topics of 'Introduction to AI Engineering with Python' HyperSkill course.
#### Learn more at:
https://hyperskill.org/projects/514
