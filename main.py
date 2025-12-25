import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from utils.retrieval import format_docs, retrieve_chain_without_lcel

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)


def retrieve_chain_with_lcel():
    """
    Create a retrieval chain with LCEL.
    Returns a chain that can be invoked with ('question': '...')

    Advantages over non-LCEL implementation:
    - No need to manually retrieve documents, format them, and pass them to the LLM
    - More efficient, as the chain is optimized for retrieval and prompt formatting
    - More readable, as the chain is written in a more declarative style
    - More modular, as the chain can be easily reused in other projects
    - More scalable, as the chain can be easily parallelized
    - More robust, as the chain can handle errors gracefully
    - More maintainable, as the chain is written in a more readable format
    - More testable, as the chain can be easily tested


    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: x["question"]) | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is pinecone in machine learning?"

    ##############################################################################################################################
    # IMPLEMENTATION 1: Raw LLM Invocation (No RAG)
    ##############################################################################################################################

    print("\n" + "-" * 70)
    print("IMPLEMENTATION 1: Raw LLM Invocation (No RAG)")
    print("-" * 70)

    result_raw_llm = llm.invoke([HumanMessage(content=query)])
    print("\n\nResponse:\n")
    print(result_raw_llm.content)

    ##############################################################################################################################
    # IMPLEMENTATION 2: Without LCEL
    ##############################################################################################################################

    print("\n" + "-" * 70)
    print("IMPLEMENTATION 2: Without LCEL")
    print("-" * 70)

    result_without_lcel = retrieve_chain_without_lcel(
        query, retriever, format_docs, prompt_template, llm
    )

    print("\n\nResponse:\n")
    print(result_without_lcel)

    ##############################################################################################################################
    # IMPLEMENTATION 3: With LCEL
    ##############################################################################################################################

    print("\n" + "-" * 70)
    print("IMPLEMENTATION 3: With LCEL")
    print("-" * 70)

    result_with_lcel = retrieve_chain_with_lcel()
    print("\n\nResponse:\n")
    print(result_with_lcel.invoke({"question": query}))
