# Architecture Documentation

## System Overview

The Emotional Podcast RAG system uses a Retrieval-Augmented Generation (RAG) architecture to provide personalized podcast recommendations.

## Component Breakdown

### 1. Data Layer (`src/data_loader.py`)

- Loads pre-processed episodes with embeddings
- Handles CSV parsing
- Embedding format conversion

### 2. Search Layer (`src/search.py`)

- Query embedding generation
- Cosine similarity calculation
- Top-K result retrieval

### 3. LLM Layer (`src/llm.py`)

- Emotional query interpretation
- Recommendation explanation generation
- Uses GPT-4o-mini for cost efficiency

### 4. Pipeline Layer (`src/rag_pipeline.py`)

- Orchestrates complete workflow:
  1. Interpret emotional context
  2. Enhanced search query
  3. Semantic search
  4. Generate explanations

### 5. API Layer (`api/main.py`)

- FastAPI REST endpoints
- Request/response validation
- Error handling

## Data Flow

```
User Query
    ↓
[API: /api/search]
    ↓
[LLM: Emotional Interpretation]
    ↓
[Search: Enhanced Query → Embedding → Similarity]
    ↓
[LLM: Generate Explanations]
    ↓
[API: Return JSON Response]
```

## Scalability Considerations

- **Caching**: Add Redis for repeated queries
- **Batch Processing**: Process multiple queries in parallel
- **Database**: Move from CSV to PostgreSQL with pgvector
- **Monitoring**: Add logging and metrics
