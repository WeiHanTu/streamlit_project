"""
LLM Question-Answering Application

A Streamlit app for uploading documents, creating OpenAI embeddings, and performing LLM-based question answering.
"""

import os
import streamlit as st
from typing import List, Optional, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_document(file: str) -> Optional[List[Any]]:
    """
    Load a document and return its data as a list of LangChain Document objects.

    Args:
        file (str): The path to the document file.

    Returns:
        Optional[List[Any]]: List of LangChain Document objects, or None if unsupported format.
    """
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_data(data: List[Any], chunk_size: int = 256, chunk_overlap: int = 20) -> List[Any]:
    """
    Split data into chunks using RecursiveCharacterTextSplitter.

    Args:
        data (List[Any]): List of LangChain Document objects.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Any]: List of chunked documents.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks: List[Any]) -> Chroma:
    """
    Create embeddings using OpenAI and store them in a Chroma vector store.

    Args:
        chunks (List[Any]): List of chunked documents.

    Returns:
        Chroma: The vector store containing embeddings.
    """
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    return vector_store


def ask_and_get_answer(vector_store: Chroma, q: str, k: int = 3) -> str:
    """
    Query the vector store and return the LLM's answer.

    Args:
        vector_store (Chroma): The Chroma vector store.
        q (str): The user's question.
        k (int): Number of top similar chunks to retrieve.

    Returns:
        str: The answer from the LLM.
    """
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']


def calculate_embedding_cost(texts: List[Any]) -> tuple[int, float]:
    """
    Calculate the total tokens and estimated embedding cost for a list of documents.

    Args:
        texts (List[Any]): List of chunked documents.

    Returns:
        tuple[int, float]: Total tokens and estimated cost in USD.
    """
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.00002


def clear_history() -> None:
    """
    Clear the chat history from Streamlit session state.
    """
    if 'history' in st.session_state:
        del st.session_state['history']


def main() -> None:
    """
    Main entry point for the Streamlit app.
    """
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                if data is None:
                    st.stop()
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file:')
    if q:
        standard_answer = ("Answer only based on the text you received as input. "
                          "Don't search external sources. If you can't answer then return `I DONT KNOW`.")
        q_prompt = f"{q} {standard_answer}"
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q_prompt, k)
            st.text_area('LLM Answer: ', value=answer)
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q}\nA: {answer}'
            st.session_state.history = f'{value} \n {'-' * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)


if __name__ == "__main__":
    main()

