from src.search      import semantic_search
from src.llm_integeration import interpret_emotional_query, generate_explanation
from src.memory      import ConversationMemory, Turn
from src.config      import DEFAULT_TOP_K


def run_pipeline(
    user_query: str,
    collection,
    memory:     ConversationMemory,
    top_k:      int = DEFAULT_TOP_K,
) -> dict:
    # Step 1: memory context → LLM knows what was discussed before
    memory_context = memory.build_context_string()

    # Step 2: interpret emotional query
    emotional_context = interpret_emotional_query(user_query, memory_context)

    # Step 3: enhanced search (original query + LLM keywords)
    keywords       = ' '.join(emotional_context.get('search_keywords', []))
    enhanced_query = f"{user_query} {keywords}".strip()
    results        = semantic_search(enhanced_query, collection, top_k=top_k)

    # Step 4: generate explanations
    recommendations = []
    for result in results:
        explanation = generate_explanation(user_query, result, emotional_context)
        recommendations.append({
            **result,
            'explanation':      explanation,
            'emotional_context': emotional_context,
        })

    # Step 5: store turn in memory
    summary = (
        f"Found {len(recommendations)} episodes for "
        f"'{emotional_context['primary_emotion']}' — "
        + ', '.join(r['metadata']['episode_title'] for r in recommendations[:2])
    )

    memory.add(Turn(
        user_query        = user_query,
        primary_emotion   = emotional_context['primary_emotion'],
        recommendations   = recommendations,
        assistant_summary = summary,
    ))

    return {
        'query':             user_query,
        'emotional_context': emotional_context,
        'recommendations':   recommendations,
        'summary':           summary,
    }


def print_results(output: dict) -> None:
    """Pretty-print pipeline output to terminal / notebook."""
    ctx  = output['emotional_context']
    recs = output['recommendations']

    print("=" * 70)
    print("🎧  EMOTIONAL PODCAST DISCOVERY")
    print("=" * 70)
    print(f"\n📝  Query    : {output['query']}")
    print(f"🧠  Emotion  : {ctx['primary_emotion']}")
    print(f"📍  Situation: {ctx['situation']}")
    print(f"💡  Needs    : {', '.join(ctx['underlying_needs'][:2])}")
    print(f"\n✨  {len(recs)} episode(s) found\n")
    print("─" * 70)

    for i, rec in enumerate(recs, 1):
        m = rec['metadata']
        print(f"\n{i}. {m['episode_title']}")
        print(f"   Show     : {m['show_name']}")
        print(f"   Duration : {m['duration_mins']} mins")
        print(f"   Match    : {rec['similarity']:.1%}")
        print(f"\n   💬 Why this helps:")
        print(f"   {rec['explanation']}")
        print(f"\n   🔗 {m['url']}")
        print("─" * 70)
