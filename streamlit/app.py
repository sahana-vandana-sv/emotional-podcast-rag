# streamlit_app.py
# -----------------------------------------------------------
# Streamlit UI with timestamp display
# -----------------------------------------------------------
# Get absolute path to project root
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
from src.vector_store_chunked import get_chunked_collection
from scripts.rag_pipeline import run_pipeline
from src.memory import ConversationMemory
from src.data_loader import load_episodes

st.set_page_config(page_title="Emotional Podcast Discovery", page_icon="🎧")

# Initialize
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationMemory()

if 'collection' not in st.session_state:
    st.session_state.collection = get_chunked_collection()

if 'df' not in st.session_state:
    st.session_state.df = load_episodes()

# Header
st.title("🎧 Emotional Podcast Discovery")
st.caption("Find the exact moment in a podcast that addresses how you feel")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    search_method = st.selectbox(
        "Search Method",
        ["hybrid", "semantic"],
        index=0,
    )
    
    if search_method == "hybrid":
        semantic_weight = st.slider(
            "Semantic Weight",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher = more semantic, Lower = more keyword-based"
        )
    else:
        semantic_weight = 1.0
    
    top_k = st.slider("Chunks to Retrieve", 5, 20, 10)
    top_episodes = st.slider("Episodes to Show", 1, 5, 3)
    
    if st.button("🗑️ Clear Memory"):
        st.session_state.memory.clear()
        st.success("Memory cleared!")

# Main query input
user_query = st.text_area(
    "How are you feeling?",
    placeholder="e.g., I keep beating myself up over mistakes I made at work",
    height=100,
)

if st.button("🔍 Find Episodes", type="primary"):
    if user_query:
        with st.spinner("Searching for relevant moments..."):
            output = run_pipeline(
                user_query=user_query,
                collection=st.session_state.collection,
                memory=st.session_state.memory,
                df=st.session_state.df,
                top_k=top_k,
                top_episodes=top_episodes,
                search_method=search_method,
                semantic_weight=semantic_weight,
            )
        
        # Display results
        st.divider()
        
        st.subheader(f"💭 Your Query: {output['query']}")
        st.caption(f"Detected emotion: **{output['emotional_context']['primary_emotion']}**")
        
        st.info(f"Found **{len(output['episodes'])} episodes** with **{output['total_chunks_retrieved']} relevant segments**")
        
        # Display each episode
        for i, episode in enumerate(output['episodes'], 1):
            with st.container():
                st.markdown(f"### {i}. 🎧 {episode['episode_title']}")
                # Try both possible key names
                channel = episode.get('youtube_channel') or episode.get('show_name', 'Unknown')
                st.caption(f"by {channel} • Match Score: {episode['best_score']:.3f}")

                # Explanation
                st.info(f"💡 **Why this helps:** {episode['explanation']}")
                
                # Chunks/Segments
                st.markdown(f"**📍 Relevant Segments ({episode['total_chunks_retrieved']}):**")
                
                for j, chunk in enumerate(episode['chunks'], 1):
                    meta = chunk['metadata']
                    
                    with st.expander(
                        f"⏱️  {meta['start_time_display']} - {meta['end_time_display']} "
                        f"({meta['duration_display']}) • Score: {chunk['similarity']:.3f}",
                        expanded=(j == 1)  # Expand first chunk
                    ):
                        # Preview text
                        st.markdown(chunk['preview'])
                        
                        # YouTube link with timestamp
                        video_url = f"{episode['url']}&t={meta['start_time_seconds']}s"
                        st.markdown(f"[▶️ Play from {meta['start_time_display']}]({video_url})")
                
                st.divider()
    else:
        st.warning("Please enter how you're feeling")

# Show conversation history
if len(st.session_state.memory) > 0:
    with st.expander("💬 Conversation History"):
        for turn in st.session_state.memory._history:
            st.markdown(f"**You:** {turn.user_query}")
            st.markdown(f"**Emotion:** {turn.primary_emotion}")
            st.markdown(f"**Found:** {', '.join(turn.recommendations[:3])}")
            st.markdown("---")