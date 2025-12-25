import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Loading documents into vector store...")

    loader = TextLoader("mediumblog1.txt")
    documents = loader.load()
    # print(documents)
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    # Embedding is the process of converting text data into numerical vectors,
    # allowing machine learning models to work with textual information in a mathematical way.
    # Here, we use OpenAIEmbeddings to generate those embeddings for our text chunks.
    # Create an embedding model (not the actual embeddings yet)
    # This model will be used by from_documents() to convert texts into vectors
    print("Creating embedding model...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    print("Ingesting chunks into vector store...")
    # from_documents() needs both:
    # - texts: the document chunks to store (content + metadata)
    # - embeddings: the embedding model to convert texts into numerical vectors
    # The method will internally use the embedding model to generate vectors from texts
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Vector store created successfully")
