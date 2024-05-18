import requests
import streamlit as st
from scripts.helpers import search_similarity

st.title("Chat-with-your-docs")

# Sidebar
st.sidebar.header("Crawling Parameters")
start_url = st.sidebar.text_input("Start URL")
max_depth = st.sidebar.number_input("Max Depth", value=5, min_value=1)

# Similarity search
st.header("Similarity Search")

query = st.text_input("Enter your query")
openai_api_key = st.text_input("OpenAI API Key")

if st.button("Search"):
    answer = search_similarity(query, openai_api_key)
    st.text("Answer:")
    st.write(answer)

# Start crawling
if st.sidebar.button("Start Crawling"):
    st.sidebar.text("Crawling in progress...")

    api_url = "http://localhost:8000/getLinks"
    params = {"start_url": start_url, "max_depth": max_depth}
    response = requests.get(api_url, params=params)

    st.sidebar.text("Crawling completed!")
# Summarize and split documents
if st.button("Process Documents"):
    st.text("Processing documents...")

    # Make a request to FastAPI route /getText with the specified file path
    response_text = requests.get(
        "http://localhost:8000/getText", params={"path": "data/output_links.json"}
    ).json()
    st.text(response_text["status"])

    st.sidebar.text("Splitting documents...")

    # Make a request to FastAPI route /initializeDB
    response_init_db = requests.get("http://localhost:8000/initializeDB").json()
    st.sidebar.text(response_init_db["status"])

    st.text("Documents processed successfully!")
