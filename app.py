# app.py

import os
import socket
import subprocess
import time
import random
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from personaagent.agent import ask, create_history
from personaagent.config import Config
from personaagent.models import create_llm
from personaagent.tools import get_available_tools, with_sql_cursor

def ensure_ollama_server(model_name: str = "qwen3:1.7b", port: int = 11434):
    """Make sure the Ollama server is running; if not, start it in background."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return
    except OSError:
        subprocess.Popen(
            ["ollama", "run", model_name, "--serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=(os.name == "nt"),
        )
        # wait up to ~10 seconds for the server to come up
        for _ in range(20):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"Failed to start Ollama server on port {port}")

# ensure Ollama is serving before anything else
ensure_ollama_server()

LOADING_MESSAGES = [
    "Consulting the ancient tomes of SQL wisdom...",
    "Casting query spells on your database...",
    "Summoning data from the digital realms...",
    "Deciphering your request into database runes...",
    "Brewing a potion of perfect query syntax...",
    "Channeling the power of database magic...",
    "Translating your words into the language of tables...",
    "Waving my SQL wand to fetch your results...",
    "Performing database divination...",
    "Aligning the database stars for optimal results...",
    "Consulting with the database spirits...",
    "Transforming natural language into database incantations...",
    "Peering into the crystal ball of your database...",
    "Opening a portal to your data dimension...",
    "Enchanting your request with SQL magic...",
    "Invoking the ancient art of query optimization...",
    "Reading between the tables to find your answer...",
    "Conjuring insights from your database depths...",
    "Weaving a tapestry of joins and filters...",
    "Preparing a feast of data for your consideration...",
]


@st.cache_resource(show_spinner=False)
def get_model() -> BaseChatModel:
    llm = create_llm()                       # <-- panggil tanpa argumen
    return llm.bind_tools(get_available_tools())



def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="ObrolDB",
    page_icon="🧙‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_css("assets/style.css")

st.header("ObrolDB")
st.subheader("Sini ngobrol sama data yang lu punya.!!!")

with st.sidebar:
    st.write("# Database Information")
    st.write(f"**File:** {Config.Path.DATABASE_PATH.relative_to(Config.Path.APP_HOME)}")
    db_size = Config.Path.DATABASE_PATH.stat().st_size / (1024 * 1024)
    st.write(f"**Size:** {db_size:.2f} MB")

    with with_sql_cursor() as cursor:
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        st.write("**Tables:**")
        for table in tables:
            # escape dan quote nama tabel
            safe_table = table.replace('"', '""')
            cursor.execute(f'SELECT count(*) FROM "{safe_table}";')
            count = cursor.fetchone()[0]
            st.write(f"- {table} ({count} rows)")

if "messages" not in st.session_state:
    st.session_state.messages = create_history()

for message in st.session_state.messages:
    if type(message) is SystemMessage:
        continue
    is_user = type(message) is HumanMessage
    avatar = "🐧" if is_user else "🤖"
    with st.chat_message("user" if is_user else "ai", avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Masukkan pertanyaan Anda..."):
    with st.chat_message("user", avatar="🐧"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(random.choice(LOADING_MESSAGES)):
            response = ask(prompt, st.session_state.messages, get_model())
            st.markdown(response)
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.session_state.messages.append(AIMessage(content=response))
