import streamlit as st
import requests
import os

# --- Configuration ---
# The backend API is running on localhost port 8000
API_BASE_URL = "http://127.0.0.1:8000"

# --- Helper Functions to Call API ---

def api_signup(username, password):
    """Calls the /signup endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/signup",
            json={"username": username, "password": password}
        )
        return response
    except requests.exceptions.ConnectionError:
        return None

def api_login(username, password):
    """Calls the /login endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            json={"username": username, "password": password}
        )
        return response
    except requests.exceptions.ConnectionError:
        return None

def api_ask(username, question):
    """Calls the /ask endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"username": username, "question": question}
        )
        return response
    except requests.exceptions.ConnectionError:
        return None


# --- Streamlit UI ---

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📄 RAG Model Chat Application")

# --- Session State Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.chat_history = [] # Each item will be a dict {'user': '...', 'assistant': '...'}

# --- Authentication UI ---
if not st.session_state.logged_in:
    st.header("Login or Signup")
    auth_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with auth_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            login_button = st.form_submit_button("Login")

            if login_button:
                if not login_username or not login_password:
                    st.error("Please enter both username and password.")
                else:
                    response = api_login(login_username, login_password)
                    if response is None:
                        st.error("Could not connect to the backend. Is it running?")
                    elif response.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.success(response.json().get("message"))
                        st.rerun()
                    else:
                        st.error(f"Failed to login: {response.json().get('detail')}")

    with signup_tab:
        with st.form("signup_form"):
            signup_username = st.text_input("Choose a Username", key="signup_user")
            signup_password = st.text_input("Choose a Password", type="password", key="signup_pass")
            signup_button = st.form_submit_button("Sign Up")

            if signup_button:
                if not signup_username or not signup_password:
                    st.error("Please enter both username and password.")
                else:
                    response = api_signup(signup_username, signup_password)
                    if response is None:
                        st.error("Could not connect to the backend. Is it running?")
                    elif response.status_code == 200:
                        st.success(response.json().get("message"))
                        st.info("You can now log in with your new credentials.")
                    else:
                        st.error(f"Failed to sign up: {response.json().get('detail')}")

# --- Main Chat Interface ---
else:
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        # A proper logout would invalidate the session on the backend
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.chat_history = []
        st.rerun()

    st.header("Chat with your Documents")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"]["answer"])
            with st.expander("Show Sources"):
                for i, doc in enumerate(message["assistant"]["docs"]):
                    source_file = os.path.basename(doc['metadata'].get('source', 'Unknown'))
                    st.write(f"**Reference {i+1}: {source_file}**")
                    st.caption(f"{doc['page_content'][:300]}...")


    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to history and display it
        st.session_state.chat_history.append({"user": prompt, "assistant": {"answer": "...", "docs": []}})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from API
        with st.spinner("Thinking..."):
            response = api_ask(st.session_state.username, prompt)

        with st.chat_message("assistant"):
            if response is None:
                st.error("Could not connect to the backend.")
                st.session_state.chat_history.pop() # Remove user prompt if API fails
            elif response.status_code == 200:
                answer_data = response.json()
                st.session_state.chat_history[-1]["assistant"] = answer_data # Update the last message
                st.markdown(answer_data["answer"])
                with st.expander("Show Sources"):
                    for i, doc in enumerate(answer_data["docs"]):
                        source_file = os.path.basename(doc['metadata'].get('source', 'Unknown'))
                        st.write(f"**Reference {i+1}: {source_file}**")
                        st.caption(f"{doc['page_content'][:300]}...")

            else:
                st.error(f"Error: {response.json().get('detail')}")
                st.session_state.chat_history.pop() # Remove user prompt if API fails


#  streamlit run streamlit_app.py
