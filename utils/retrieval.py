"""Retrieval chain utilities for RAG implementation."""


def format_docs(docs):
    """Format the retrieved documents into a string that can be used as context for the LLM."""
    return "\n\n".join([doc.page_content for doc in docs])


def retrieve_chain_without_lcel(query: str, retriever, format_docs, prompt_template, llm):
    """
    Simple retrieval chain without LCEL.
    Manually retrieve documents, format them, and pass them to the LLM.

    Args:
        query: The query string to search for
        retriever: The retriever instance to use for document retrieval
        format_docs: Function to format retrieved documents
        prompt_template: The prompt template to use
        llm: The language model instance

    Returns:
        The response content from the LLM
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content

