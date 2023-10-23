__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import openai
import os

load_dotenv()

import openai

st.title("Instagram Post Generator")

# Input for URL
url = st.text_input("Enter the URL:")
openai_api_key = st.text_input("Openai api key")

def create_chroma_db_without_embeddings(embeddings):
    return Chroma(embedding_function=embeddings, persist_directory="./embeddings/{}-embeds".format('xlsx'))

if url and openai_api_key:
    OPENAI_API_KEY = openai_api_key

    urls = [url]

    loader = WebBaseLoader(urls)
    docs = loader.load()
    import shutil
    directory = 'embeddings'
    if os.path.exists(directory):
        shutil.rmtree(directory)

    print("splitting")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separators=[" ", ",", "\n"])
    texts = text_splitter.split_documents(docs)
    print("embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = Chroma.from_documents(texts, embeddings)
    print("trained successfully!")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model_name="gpt-4-0613", openai_api_key=OPENAI_API_KEY)

    qa_template = """
    You are helpful AI assistant to create instagram post based on provided context.

    context: {context}
    =========
    question: {question}
    ======
    """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT})

    response_old = qa({"query": "Please create a 300 words instagram post using provided context"})

    st.write(response_old['result'])





