import os
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "https://emotional-podcast-rag.onrender.com")

st.set_page_config(page_title="Podcast Discovery", layout="wide")
st.title("Emotional Podcast RAG")

query = st.text_input("What are you feeling now ?")
num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=3)

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        payload = {
            "query": query.strip(),
            "num_recommendations": num_recs,
        }

        r = requests.post(f"{API_BASE_URL}/api/search", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

    st.subheader("Results")
    results = data.get("results") or data.get("recommendations") or []

    for item in results:
        # adjust keys to what your backend returns
        title = item.get("episode_title") or item.get("metadata", {}).get("episode_title") or "Untitled"
        url = item.get("url") or item.get("metadata", {}).get("url") or ""
        snippet = item.get("snippet") or item.get("document") or ""

        st.write(f"**{title}**")
        if snippet:
            st.write(snippet)
        if url:
            st.write(url)
        st.divider()