import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Loading vector store...")
    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"])
    print("Vector store loaded successfully")
    print("Vector store:", vector_store)
