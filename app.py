import streamlit as st
from dotenv import load_dotenv
import RAG_util
import chat_session
import time

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.1)


def handle_userinput(user_input):
    """
    This function handles user input and updates the conversation history.
    It takes a user's question as input, sends it to the conversation chain,
    and updates the chat history with the response.

    Parameters:
    user_input (str): The user's question to be processed.

    Returns:
    None
    """
    for message in st.session_state.chat_session.history:
        with st.chat_message((message["role"])):
            st.markdown(message['content'])

def main():
    """
    The main function of the PDF Question Answer Bot using RAG.
    This function handles the user interface, document processing, and conversation flow.

    Parameters:
    None

    Returns:
    None
    """
    load_dotenv()
    st.set_page_config(
        page_title="PDF Question Answer Bot Using RAG",
        page_icon='rag.jpg',
        layout="wide"
    )

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = chat_session.chat_session()
    
    
    st.header("Upload and Chat with you text pdf and ppt files")
    
    with st.sidebar:
        st.subheader("Upload Document")
        file = st.file_uploader(
            "Upload your pdf, txt or ppt file here and  click on 'Process'", 
            accept_multiple_files=False,
            type=("pdf", "txt", "ppt", "pptx")
        )

        if(file is None):
            st.error("No file uploaded. Please upload first.")


        if (file is not None and st.button("Process")):
            with st.spinner("Processing"):

                #get and read raw document
                doc = RAG_util.read_pdf(file)

                #split document into blobs
                blobs = RAG_util.split_text(doc)

                #embade and vectorize
                vector_db = RAG_util.load_vector_db(blobs)

                llm = RAG_util.init_llm()

                st.session_state.llm = llm
                st.session_state.vdb = vector_db

    if("vdb" in st.session_state):
        user_query = st.chat_input("Ask Your Question")        
        if user_query:
            st.session_state.chat_session.history.append({'role': 'user', 'content': user_query})
            handle_userinput(user_query)
            response = RAG_util.generate_response(st.session_state.llm, st.session_state.vdb, user_query)
            st.session_state.chat_session.history.append({'role': 'assistant', 'content': response})
            with st.chat_message("assistant"):
                st.write_stream(stream_data(response))

                #initialize conversation chain
                #chat_agent = RAG_util.init_conversation_chain(vector_db)

                #st.session_state.conversation = chat_agent

                #clear chat history
                #st.session_state.chat_history = None

if __name__ == "__main__":
    main()
