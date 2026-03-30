import subprocess
import sys
from pathlib import Path

import streamlit as st
import requests
import random

from clients.base import BaseClient

# ── Streamlit UI ─────────────────────────────────────────────────────────────
# Everything below this block is executed by Streamlit on every app rerun.
# Do not call StreamlitClient().start() here — use `streamlit run` instead.

AGENT_API_URL = "http://localhost:8000/ask"

JOKES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "What do you call a fake noodle? An impasta!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What do you call cheese that isn't yours? Nacho cheese!",
    "How do you organize a space party? You planet!",
]


def _get_agent_response(user_query: str) -> str:
    with st.spinner(random.choice(JOKES)):
        try:
            response = requests.get(AGENT_API_URL, params={"prompt": user_query})
            response.raise_for_status()
            return response.json().get("response", "No response from agent.")
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to the Agent API. Please ensure the REST server is running."
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"


def _run_ui() -> None:
    st.set_page_config(layout="wide")
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
        unsafe_allow_html=True,
    )
    st.title(" Your own mini property agent 🕵️‍♂️🏡")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you find your next property?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "agent", "content": _get_agent_response(prompt)})
        st.rerun()


# ── BaseClient implementation ─────────────────────────────────────────────────

class StreamlitClient(BaseClient):
    """Launches the Streamlit UI server as a subprocess."""

    def start(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__))],
            check=True,
        )


# Executed by `streamlit run clients/streamlit/app.py`
_run_ui()
