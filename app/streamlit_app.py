import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface

st.title("Gynecological RAG Chatbot")

# Initilize session state variable
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Displaying sidebar
display_sidebar()

# Displaying chat-interface
display_chat_interface()

