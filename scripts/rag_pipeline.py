import json
from collections import defaultdict
from src.search      import semantic_search
from src.llm_integeration import interpret_emotional_query, generate_explanation
from src.memory      import ConversationMemory, Turn
from src.config      import DEFAULT_TOP_K
from src.logging_utils import get_logger

logger = get_logger("rag_pipeline")

def group_chunks_by_episode(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    
    logger.debug(f"Grouping {len(chunks)} chunks by episode...")
    
    # Group by episode_id
    grouped = defaultdict(list)
    
    for chunk in chunks:
        episode_id = chunk['metadata']['episode_id']
        grouped[episode_id].append(chunk)
    
    # Convert to list of episodes with metadata
    episodes = []
    
    for episode_id, episode_chunks in grouped.items():
        
        # Sort chunks by relevance (highest score first)
        sorted_chunks = sorted(
            episode_chunks,
            key=lambda x: x['similarity'],
            reverse=True
        )
        
        # Get episode metadata from first chunk
        first_chunk = sorted_chunks[0]
        meta = first_chunk['metadata']
        
        episodes.append({
            'episode_id': episode_id,
            'episode_title': meta['episode_title'],
            'show_name': meta['youtube_channel'],
            'url': meta['url'],
            'video_id': meta.get('video_id', ''),
            'best_score': sorted_chunks[0]['similarity'],
            'chunks': sorted_chunks,
            'total_chunks_retrieved': len(sorted_chunks),
        })
    
    # Sort episodes by best chunk score
    episodes.sort(key=lambda x: x['best_score'], reverse=True)
    
    logger.info(
        f"Grouped into {len(episodes)} episodes "
        f"(avg {sum(e['total_chunks_retrieved'] for e in episodes) / len(episodes):.1f} chunks per episode)"
    )
    
    return episodes

def run_pipeline(
    user_query: str,
    collection,
    memory:     ConversationMemory,
    df,
    top_k: int = 10,  # Retrieve more chunks initially
    top_episodes: int = 3,  # Show top 3 episodes in final results
    search_method: str = "hybrid",  # "semantic" or "hybrid"
    semantic_weight: float = 0.8, # fine tuned weight 
) -> dict:
    logger.info(f"Pipeline started: '{user_query[:50]}...'")

    # Step 1: memory context → LLM knows what was discussed before
    memory_context = memory.build_context_string()

    # Step 2: interpret emotional query
    emotional_context = interpret_emotional_query(user_query, memory_context)

    # Step 3: enhanced search (original query + LLM keywords)
    keywords       = ' '.join(emotional_context.get('search_keywords', []))
    enhanced_query = f"{user_query} {keywords}".strip()
    if search_method == 'hybrid':
        from src.hybrid_search_chunked import HybridSearcherChunked
        searcher = HybridSearcherChunked(collection)  # ← Uses chunked collection directly
        chunks = searcher.search(
            query=enhanced_query,
            top_k=top_k,
            semantic_weight=semantic_weight,
        )
    else:
        chunks = semantic_search(query=enhanced_query, collection=collection,
            top_k=top_k)

    logger.info(f"Pipeline started: '{user_query[:50]}...'")

    #Group chunks by episode
    episodes = group_chunks_by_episode(chunks)
    # Take top N episodes
    episodes = episodes[:top_episodes]
    
    logger.info(f"Returning top {len(episodes)} episodes")

    # Step 4: generate explanations
    logger.debug("Generating explanations...")
    for episode in episodes:
        
        # Use the best (highest-scoring) chunk for explanation context
        best_chunk = episode['chunks'][0]

        # Create explanation input that matches what generate_explanation expects
        explanation_input = {
        'metadata': {
            'episode_title': episode['episode_title'],
            'show_name': episode['show_name'],  # ← This key was missing
            'youtube_channel': episode['show_name'],  # Include both for compatibility
        },
        'preview': best_chunk['preview'],
    }
        explanation = generate_explanation(user_query, episode_result=explanation_input, emotional_context=emotional_context)
        episode['explanation'] = explanation

    # Step 5: store turn in memory
    summary = [
        f"{ep['episode_title']} ({ep['total_chunks_retrieved']} segments)"
        for ep in episodes
    ]

    memory.add(Turn(
        user_query        = user_query,
        primary_emotion   = emotional_context['primary_emotion'],
        recommendations   = summary,
        assistant_summary = f"Found {len(episodes)} episodes...",
    ))

    return {
        'query':             user_query,
        'emotional_context': emotional_context,
        'episodes': episodes,
        'total_chunks_retrieved': len(chunks),
        'search_method': search_method,
        'semantic_weight': semantic_weight if search_method == "hybrid" else 1.0,
    }


def print_results(output: dict) -> None:
    """
    Pretty-print pipeline results with timestamps.
    
    Parameters
    ----------
    output : Pipeline output dict
    """
    print("\n" + "="*60)
    print("🎯 EMOTIONAL PODCAST RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n💭 Your Query: {output['query']}")
    print(f"😢 Detected Emotion: {output['emotional_context']['primary_emotion']}")
    print(f"🔍 Search Method: {output['search_method']}")
    
    if output['search_method'] == 'hybrid':
        print(f"⚖️  Weights: {output['semantic_weight']:.0%} semantic, {1-output['semantic_weight']:.0%} keyword")
    
    print(f"\n📊 Found {len(output['episodes'])} episodes with {output['total_chunks_retrieved']} relevant segments")
    
    for i, episode in enumerate(output['episodes'], 1):
        print(f"\n{'─'*60}")
        print(f"{i}. 🎧 {episode['episode_title']}")
        print(f"   by {episode['show_name']}")
        print(f"   Match Score: {episode['best_score']:.3f}")
        
        print(f"\n   📍 Relevant Segments ({episode['total_chunks_retrieved']}):")
        
        for j, chunk in enumerate(episode['chunks'], 1):
            meta = chunk['metadata']
            print(f"\n   {j}. ⏱️  {meta['start_time_display']} - {meta['end_time_display']} ({meta['duration_display']})")
            print(f"      Score: {chunk['similarity']:.3f}")
            print(f"      Preview: {chunk['preview'][:100]}...")
        
        print(f"\n   💡 Why this helps:")
        print(f"      {episode['explanation']}")
        
        print(f"\n   🔗 Watch: {episode['url']}&t={episode['chunks'][0]['metadata']['start_time_seconds']}s")
    
    print("\n" + "="*60)
