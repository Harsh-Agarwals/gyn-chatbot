# Importing all the necessary libraries
# Some of the important libraries includes:
# - langchain (for making application)
# - pinecone (for database)
# - sqlite3 (for storing responses in tables)

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

from dotenv import load_dotenv
import os

# Extracting API Keys for Openai and pinecone
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_data(path):
    """
    Function to load pdf data 
    Input: path = path to the directory containing PDFs
    Output: documents = loaded pdfs
    """
    dir_loader = DirectoryLoader(
        path=path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = dir_loader.load()
    return documents

def text_splitter(docs):
    """
    Function that split the documents into chunks
    Input: docs = loaded documents 
    Output: splits = splitted chunks of data
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    return splits

def define_embeddings():
    """
    Defining embedding method here.
    I am using HuggingFaceEmbedding here, as it is free to use and is easy.
    Output: embedding model
    """
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
    """
    Initializing pinecone database and making an index if it does not exists
    Input: Index name
    Output: Nothing, it just connects to db and initializes the pinecone database with the given index
    """
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
    """
    This function initialize the retriever and returns it
    10 closest retrievers are outputted so as the LLM produces a good results during the RAG
    Search method: Similarity score (Cosing score)
    Input: index_name = index in pinecone db where the embeddings are to be stores
    Output: retriever = returns the retriever after saving embeddings in the db
    """
    embd = define_embeddings()
    # docSearch = PineconeVectorStore.from_documents(
    #     documents=data_split,
    #     index_name=index_name,
    #     embedding=embd
    # )
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embd
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    return retriever

def from_loading_to_embedding(path, pinecone_index_name):
    """
    This function combines all the major steps from loading the PDFs, splitting, initilizing database to getting the retriever. This function is made to make our workflow easy.
    Whenever a new data is added in the knowledge base, this code can run to save it's embeddings in the database. It also returns the retriever for the Query and RAG purpose

    Input: path = path of directory containing PDFs, pinecone_index_name = index name of the pinecone db where embeddings are to be stored
    Output: Returns retriever for pinecone db index
    """
    docs = extract_data(path)
    splits = text_splitter(docs)
    initialize_pinecone(index_name=pinecone_index_name)
    retriever = get_retriever_from_vector_store(splits, pinecone_index_name)
    return retriever

def get_llm(api_key, temperature=0.4, max_tokens=500):
    """
    Initializing and returing LLM
    I am using OpenAI for the LLM

    Input:
        - api_key: api key for OpenAI
        - temperature: temperature for output responses
        - max_tokens: maximum tokens to return while giving answer to the question
    Output:
        - LLM: outputs the large language model initialized from the above inputs
    """
    llm = OpenAI(api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    return llm

def initialize_chat_history():
    """
    Making an empty chat history here.
    This will be initialized whenever the user starts new chat.

    returns empty list
    """
    chat_history = []
    return chat_history

def extend_chat_history(chat_history, question, response):
    """
    Filling up the chat+history list with input question and LLM response
    This is helpful in tracing the history and in Conversational RAG

    Input:
        - chat_history: list (empty or containing chat history)
        - question: user question
        - response: LLM response
    Output:
        - chat_history: list of chat history with new QnA appended.
    """
    chat_history.extend([
        AIMessage(content=question),
        HumanMessage(content=response)
    ])
    return chat_history

def setup():
    """
    This function combines last few steps
    - Initializes the LLm
    - starts the session id
    - initializes the chat history

    Returns: returns all three
    """
    llm = get_llm(api_key=OPENAI_API_KEY, temperature=0.5)
    session_id = init_session_id()
    chat_history = initialize_chat_history()

    return llm, session_id, chat_history

def get_history_aware_retriever(llm, retriever):
    """
    Given a chat history and a new question referencing the last chat, it is important to re structure the question(with correct terms removing referencing words like "it", "this" etc)
    Eg: User question: "Tell me how to prevent this"
        New rephrased question: "Tell me how to prevent Ovarian Cancer"

    This function does the same

    Using LLM at its core, it gets user history and LLM, converting the question to the relevant rephrased question
    """
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
    """
    This function sets up the RAG chain

    Uses
    - create_stuff_documents_chain => creates qna chain using llm and prompt, and
    - create_retrieval_chain => creates RAG chain using history aware retriever and qna chain

    Returns: RAG chain
    """
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
    Initializes the RAG Chain

    Set up llm, session id and chat history
    Use these to create history aware retriever and RAG chain

    Output: rag_chain, session_id, chat_histor
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
    Describes the whole workflow

    - first getting retriever using from_loading_to_embedding()
    - then getting rag_chain, session_id and chat_history using rag_chain_init()
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

def invoke_rag_and_storein_db(DB_NAME, rag_chain, session_id, question, chat_history):
    """
    When user asks question, this function gets invoked.
    The response is recorded using LLM invoking and then inserted into the local database.
    Chat history gets updated.

    Input: 
        - DB_NAME: db name to store conversations
        - session_id: session id for which communication happens
        - rag_chain: RAG Chain for QnA
        - question: user query
        - chat_history: Chat history list
    Output:
        - response: LLM answer to user query
        - chat_history: updated chat history list
    """
    response = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
    insert_application_logs(DB_NAME, session_id, question, response, "gpt-40-mini")
    chat_history = get_chat_history(DB_NAME, session_id=session_id)
    return response, chat_history


def connect_db(DB_NAME):
    """
    Connecting to the new SQL db
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs(DB_NAME):
    """
    Creating the SQL table to store data
    """
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

def insert_application_logs(DB_NAME, session_id, user_query, gpt_response, model):
    """
    Inserting data in SQL table
    """
    conn = connect_db(DB_NAME)
    conn.execute('''INSERT INTO application_logs 
                 (session_id, user_query, gpt_response, model) VALUES 
                 (?, ?, ?, ?)''', (session_id, user_query, gpt_response, model))
    
    conn.commit()
    conn.close()

def get_chat_history(DB_NAME, session_id):
    """
    Retrieving data from the SQL table
    """
    conn = connect_db(DB_NAME)
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
    """
    Initializing the Database setup fir setting the db for storing chats
    """
    create_application_logs(DB_NAME)

def init_session_id():
    """
    Setting up the session id

    The session id is helpful for new chats. Every chat will have a particular session id which will be used to store and track histories in that threads.

    Returns: Unique ID
    """
    return str(uuid.uuid4())

def question_answering(DB_NAME, session_id, rag_chain, question):
    """
    User enters question
    This function gets invoked

    LLM is invoked to produced the answer and gets stored in chat_history, which in turns gets stored in local db to trace conversations.

    Input:
        - DB_NAME: db name to store conversations
        - session_id: session id for which communication happens
        - rag_chain: RAG Chain for QnA
        - question: user query

    Output:
        - answer: LLM answer to the question (This will be reflected in the frontend)
        - chat_history: Chat history list which stores conversations for historical contexts
    """
    chat_history = get_chat_history(DB_NAME, session_id=session_id)
    answer, chat_history = invoke_rag_and_storein_db(DB_NAME, rag_chain, session_id, question, chat_history)
    return answer, chat_history


