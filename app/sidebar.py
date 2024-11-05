import streamlit as st

def display_sidebar():
    # Sidebar model options
    model_options = ["gpt-4o-mini", "gpt-3.5-turbo"]
    st.sidebar.selectbox(label="Select Model", options=model_options, key="model")

    # Sidebar document upload
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader(label="Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(f"File {uploaded_file} updated successfully with ID {upload_response['file_id']}.")
                    st.session_state.documents = list_documents()

    # Sidebar listing documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()

    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents

    

