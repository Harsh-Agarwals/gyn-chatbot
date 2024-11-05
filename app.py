import streamlit as st
from src.helper import rag_chain_init, question_answering, from_loading_to_embedding

retriever = from_loading_to_embedding(path="data/", pinecone_index_name="gyne-chatbot")
rag_chain, session_id, chat_history = rag_chain_init(retriever=retriever, DB_NAME="data/gyne_app.db")

st.title("Gynecological Conversational Chatbot")

question = st.text_input(label="How can I assist you today?")
if st.button("Ask"):
    answer, chat_history = question_answering(DB_NAME="data/gyen_app.db", session_id=session_id, rag_chain=rag_chain, question=question)

    st.text(answer)
