from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import sqlite3
import uuid
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_data(path):
    dir_loader = DirectoryLoader(
        path=path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = dir_loader.load()
    return documents

def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    return splits

def define_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf

def initialize_pinecone(index_name):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = index_name
    try:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
    except Exception as e:
        pass

def get_retriever_from_vector_store(data_split, index_name):
    embd = define_embeddings()
    docSearch = PineconeVectorStore.from_documents(
        documents=data_split,
        index_name=index_name,
        embedding=embd
    )
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embd
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

def from_loading_to_embedding(path, pinecone_index_name):
    """
    Important
    """
    docs = extract_data(path)
    splits = text_splitter(docs)
    initialize_pinecone(index_name=pinecone_index_name)
    retriever = get_retriever_from_vector_store(splits, pinecone_index_name)
    return retriever

def get_llm(api_key, temperature=0.4, max_tokens=500):
    llm = OpenAI(api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    return llm

def initialize_chat_history():
    chat_history = []
    return chat_history

def extend_chat_history(chat_history, question, response):
    chat_history.extend([
        AIMessage(content=question),
        HumanMessage(content=response)
    ])
    return chat_history

def setup():
    """
    Important
    """
    llm = get_llm(api_key=OPENAI_API_KEY, temperature=0.5)
    session_id = init_session_id()
    chat_history = initialize_chat_history()

    return llm, session_id, chat_history

def get_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    contextualized_chain = contextualize_q_prompt | llm | StrOutputParser()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def setup_rag_chain(llm, history_aware_retriever):
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an assistant for gynaecological tasks for question answering.
                Remeber your answers are very critical for a person.
                Answer the question as asked by the user in the most precise manner. Use the 
                following pieces of retrieved context to answer the questions. If you don't know the
                answer, clearly say you dont know. Use three sentence maximum and keep the answers 
                concise, crisp and clear. Use the following context and answer the question."""),
            ("system", "Context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def rag_chain_init(retriever, DB_NAME):
    """
    Important
    """
    llm, session_id, chat_history = setup()
    history_aware_retriever = get_history_aware_retriever(llm, retriever)
    rag_chain = setup_rag_chain(llm, history_aware_retriever)
    try:
        init_application_logs(DB_NAME)
    except Exception as e:
        pass
    return rag_chain, session_id, chat_history

def workflow(path, pinecone_index_name, DB_NAME):
    """
    Important
    """
    # Retriever to be invoked as soon as the app is started
    retriever = from_loading_to_embedding(path, pinecone_index_name)
    # Below will be invoked as soon as we start a new chat
    rag_chain, session_id, chat_history = rag_chain_init(retriever, DB_NAME)
    # Everythin is now set. User will now question and we will get answer using rag_chain. Also these will get stored in chat history of this session id.

def invoke_rag_and_history(rag_chain, question, chat_history):
    response = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
    chat_history = extend_chat_history(chat_history, question, response)
    return response, chat_history

def invoke_rag_and_storein_db(rag_chain, session_id, question, chat_history):
    response = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
    insert_application_logs(session_id, question, response, "gpt-40-mini")
    chat_history = get_chat_history(session_id=session_id)
    return response, chat_history


def connect_db(DB_NAME):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs(DB_NAME):
    conn = connect_db(DB_NAME)
    conn.execute('DROP TABLE IF EXISTS application_logs')
    conn.execute('''CREATE TABLE application_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_id TEXT,
                 user_query TEXT,
                 gpt_response TEXT,
                 model TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = connect_db()
    conn.execute('''INSERT INTO application_logs 
                 (session_id, user_query, gpt_response, model) VALUES 
                 (?, ?, ?, ?)''', (session_id, user_query, gpt_response, model))
    
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    except Exception as e:
        print(f"Error: {e}")
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

def init_application_logs(DB_NAME):
    create_application_logs(DB_NAME)

def init_session_id():
    return uuid.uuid4()

def question_answering(session_id, rag_chain, question):
    """
    Important
    """
    chat_history = get_chat_history(session_id=session_id)
    answer, chat_history = invoke_rag_and_storein_db(rag_chain, session_id, question, chat_history)
    return answer, chat_history


