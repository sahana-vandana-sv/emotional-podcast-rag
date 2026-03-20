# streamlit_app.py
# -----------------------------------------------------------
# Streamlit UI with timestamp display and feedback collection
# -----------------------------------------------------------

# Get absolute path to project root
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import json
from datetime import datetime
from src.vector_store_chunked import get_chunked_collection
from scripts.rag_pipeline import run_pipeline
from src.memory import ConversationMemory
from src.data_loader import load_episodes

st.set_page_config(page_title="Emotional Podcast Discovery", page_icon="🎧")

# ═══════════════════════════════════════════════════════════
# Helper Functions (defined at top level)
# ═══════════════════════════════════════════════════════════

def save_feedback(query, output, rating, comment=None):
    """Save overall query feedback to JSONL file."""
    
    # Create directory if it doesn't exist
    os.makedirs('data/evaluation', exist_ok=True)
    
    feedback = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'rating': rating,
        'comment': comment,
        'search_method': output.get('search_method', 'unknown'),
        'semantic_weight': output.get('semantic_weight', None),
        'episodes_shown': [ep['episode_id'] for ep in output['episodes']],
        'top_chunks': [
            {
                'chunk_id': ep['chunks'][0]['metadata']['chunk_id'],
                'episode_id': ep['episode_id'],
                'score': ep['chunks'][0]['similarity'],
            }
            for ep in output['episodes']
        ],
    }
    
    # Append to feedback log
    with open('data/evaluation/user_feedback.jsonl', 'a') as f:
        f.write(json.dumps(feedback) + '\n')


def save_episode_feedback(query, episode_id, chunk_id, is_relevant):
    """Save per-episode feedback to JSONL file."""
    
    os.makedirs('data/evaluation', exist_ok=True)
    
    feedback = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'episode_id': episode_id,
        'chunk_id': chunk_id,
        'is_relevant': is_relevant,
    }
    
    # Append to episode feedback log
    with open('data/evaluation/episode_feedback.jsonl', 'a') as f:
        f.write(json.dumps(feedback) + '\n')


# ═══════════════════════════════════════════════════════════
# Initialize Session State
# ═══════════════════════════════════════════════════════════

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationMemory()

if 'collection' not in st.session_state:
    st.session_state.collection = get_chunked_collection()

if 'df' not in st.session_state:
    st.session_state.df = load_episodes()

if 'current_output' not in st.session_state:
    st.session_state.current_output = None

if 'current_query' not in st.session_state:
    st.session_state.current_query = None

if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# ═══════════════════════════════════════════════════════════
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
    
    st.divider()
    
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
        # Reset feedback state for new query
        st.session_state.feedback_submitted = False
        
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
        
        # Store in session state
        st.session_state.current_output = output
        st.session_state.current_query = user_query
        
    else:
        st.warning("Please enter how you're feeling")

# ═══════════════════════════════════════════════════════════
# Display Results
# ═══════════════════════════════════════════════════════════

if st.session_state.current_output is not None:
    output = st.session_state.current_output
    
    st.divider()
    
    st.subheader(f"💭 Your Query: {output['query']}")
    st.caption(f"Detected emotion: **{output['emotional_context']['primary_emotion']}**")
    
    st.info(f"Found **{len(output['episodes'])} episodes** with **{output['total_chunks_retrieved']} relevant segments**")
    
    # Display each episode
    for i, episode in enumerate(output['episodes'], 1):
        with st.container():
            st.markdown(f"### {i}. 🎧 {episode['episode_title']}")
            
            # Channel name (try both keys)
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
                    expanded=(j == 1)
                ):
                    # Preview text
                    st.markdown(chunk['preview'])
                    
                    # YouTube link with timestamp
                    video_url = f"{episode['url']}&t={meta['start_time_seconds']}s"
                    st.markdown(f"[▶️ Play from {meta['start_time_display']}]({video_url})")
            
            st.divider()
    
    # ═══════════════════════════════════════════════════════════
    # Feedback Section
    # ═══════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("💭 How helpful were these recommendations?")
    
    # Overall feedback
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Very helpful", key="very_helpful", use_container_width=True):
            save_feedback(
                st.session_state.current_query, 
                output, 
                rating=5
            )
            st.session_state.feedback_submitted = True
            st.success("Thank you for your feedback!")
    
    with col2:
        if st.button("😐 Somewhat helpful", key="somewhat_helpful", use_container_width=True):
            save_feedback(
                st.session_state.current_query, 
                output, 
                rating=3
            )
            st.session_state.feedback_submitted = True
            st.success("Thank you! We'll keep improving.")
    
    with col3:
        if st.button("👎 Not helpful", key="not_helpful", use_container_width=True):
            st.session_state.show_comment_box = True
    
    # Comment box for negative feedback
    if st.session_state.get('show_comment_box', False):
        with st.form("negative_feedback_form"):
            st.markdown("**What could be better?**")
            comment = st.text_area(
                "Your feedback helps us improve",
                placeholder="e.g., Results were too generic, missing specific advice, etc.",
                height=100,
            )
            
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                save_feedback(
                    st.session_state.current_query,
                    output,
                    rating=1,
                    comment=comment
                )
                st.session_state.feedback_submitted = True
                st.session_state.show_comment_box = False
                st.success("Thank you for helping us improve!")
                st.rerun()
    
    # Per-episode feedback
    st.markdown("### 📊 Rate Individual Episodes")
    st.caption("Help us understand which recommendations were most relevant")
    
    for i, episode in enumerate(output['episodes']):
        with st.expander(
            f"Episode {i+1}: {episode['episode_title'][:50]}...",
            expanded=False
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.caption(f"**Top segment:** {episode['chunks'][0]['metadata']['start_time_display']} - {episode['chunks'][0]['metadata']['end_time_display']}")
                st.caption(f"**Score:** {episode['best_score']:.3f}")
            
            with col2:
                # Use unique key for each radio button
                rating_key = f"episode_rating_{i}_{episode['episode_id']}"
                rating = st.radio(
                    "Relevant?",
                    ["Yes", "No", "Not sure"],
                    key=rating_key,
                    horizontal=True,
                    index=2,  # Default to "Not sure"
                )
            
            # Save when user selects Yes or No
            if rating in ["Yes", "No"]:
                save_episode_feedback(
                    st.session_state.current_query,
                    episode['episode_id'],
                    episode['chunks'][0]['metadata']['chunk_id'],
                    is_relevant=(rating == "Yes")
                )
                
                if rating == "Yes":
                    st.success("✓ Marked as relevant")
                else:
                    st.info("✗ Marked as not relevant")
    
    # Show thank you message if feedback submitted
    if st.session_state.feedback_submitted:
        st.balloons()
        st.success("🙏 Thank you for your feedback! Your input helps improve the system.")

# ═══════════════════════════════════════════════════════════
# Conversation History
# ═══════════════════════════════════════════════════════════

if len(st.session_state.memory) > 0:
    with st.expander("💬 Conversation History", expanded=False):
        # Access turns using get_turns() method
        try:
            turns = st.session_state.memory.get_turns()
        except AttributeError:
            # Fallback to internal attribute if method doesn't exist
            turns = st.session_state.memory._history
        
        for i, turn in enumerate(turns, 1):
            st.markdown(f"**Query {i}:** {turn.user_query}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.caption(f"Emotion: {turn.primary_emotion}")
            with col2:
                recs = turn.recommendations[:3]
                st.caption(f"Found: {', '.join(recs)}")
            
            if i < len(turns):
                st.markdown("---")

# ═══════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.caption("💡 Tip: Try being specific about how you're feeling for better recommendations")