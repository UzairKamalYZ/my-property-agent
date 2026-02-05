import streamlit as st
import requests
import json
import random

# Configuration for the Agent REST API
AGENT_API_URL = "http://localhost:8000/ask"

st.set_page_config(layout="wide") # Set page layout to wide

# Custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.rawpixel.com/image/5908905/free-apartment-image-public-domain-cc0-photo");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(" Your own mini property agent 🕵️‍♂️🏡")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# List of jokes to display while waiting
jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "What do you call a fake noodle? An impasta!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "How do you organize a space party? You planet!",
]

# Function to get response from backend
def get_agent_response(user_query):
    # Select a random joke
    joke = random.choice(jokes)
    
    with st.spinner(joke): # Display joke while waiting
        try:
            response = requests.get(AGENT_API_URL, params={"prompt": user_query})
            response.raise_for_status() # Raise an exception for HTTP errors
            agent_response = response.json().get("response", "No response from agent.")
        except requests.exceptions.ConnectionError:
            agent_response = "Error: Could not connect to the Agent API. Please ensure the agentRest.py server is running."
        except requests.exceptions.RequestException as e:
            agent_response = f"Error: An error occurred: {e}"
    return agent_response

# Chat input widget
if prompt := st.chat_input("How can I help you find your next property (e.g., '3-bedroom apartment in Warsaw')?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Get agent response
    agent_response = get_agent_response(prompt)
    # Add agent response to chat history
    st.session_state.messages.append({"role": "agent", "content": agent_response})
    # Rerun the app to display new messages
    st.rerun()




