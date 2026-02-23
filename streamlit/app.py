import os
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "https://emotional-podcast-rag.onrender.com")

st.set_page_config(page_title="Podcast Discovery", layout="wide")
st.title("Emotional Podcast RAG")

query = st.text_input("What do you want to learn about? (e.g., shame, self-compassion)")

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        r = requests.post(f"{API_BASE_URL}/search", json={"query": query}, timeout=60)
        r.raise_for_status()
        data = r.json()

    st.subheader("Results")
    for item in data.get("results", []):
        st.write(f"**{item.get('episode_title', 'Untitled')}**")
        st.write(item.get("snippet", ""))
        st.write(item.get("url", ""))
        st.divider()