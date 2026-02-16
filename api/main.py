#fast api is already installed 
import sys
sys.path.append('..')


from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

from src.data_loader import load_episodes
from src.vector_store import get_collection
from src.memory import ConversationMemory
from src.rag_pipeline import run_pipeline

#state
state ={
    "collection":None,
    "episodes_df":None,
    "memory":None
}

#loading data before accepting requests 
@asynccontextmanager
async def lifespan(app: FastAPI):
     # Runs ONCE when server starts
    print("Loading episodes and ChromaDB...")
    state["episodes_df"] = load_episodes()
    state["collection"]  = get_collection()
    state["memory"]      = ConversationMemory()
    print(f"✓ Ready. {len(state['episodes_df'])} episodes loaded.")
    yield
    print("shutting down")

#app
app = FastAPI(
    title="Emotional podcast RAG API",
    description="Finds podcast episodes based on how you feel.",
    version="1.0.0",
    lifespan=lifespan)
 
 # Allow frontend apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#request response schemas

class SearchRequest(BaseModel):
    query:str
    num_recommendations:Optional[int]=3

class EpisodeOut(BaseModel):
    episode_title: str
    show_name:     str
    url:           str
    duration_mins: float
    similarity:    float
    explanation:   str

class SearchResponse(BaseModel):
    query:              str
    primary_emotion:    str
    situation:          str
    recommendations:    list[EpisodeOut]
    memory_turn_count:  int

#End points

@app.get("/")
def root():
    return {
        "name":"Emotional Podcast RAG API",
        "status":"running",
        "endpoints":["/health","/api/search","/api/memory/clear"]
    }

@app.get("/health")
def health():
    return{
        "status":"healthy",
        "episodes_loaded": len(state["episodes_df"]) if state["episodes_df"] is not None else 0,
        "memory_turns":    len(state["memory"]) if state["memory"] else 0,
    }

@app.post("/api/search", response_model=SearchResponse)
def search(request:SearchRequest):
    if state['episodes_df'] is None:
        raise HTTPException(status_code='503', detail="episodes not loaded")

    try:
        output = run_pipeline(
            user_query = request.query,
            collection = state["collection"],
            memory     = state["memory"],
            top_k      = request.num_recommendations,
        )

        ctx  = output["emotional_context"]
        recs = output["recommendations"]

        return SearchResponse(
            query           = request.query,
            primary_emotion = ctx["primary_emotion"],
            situation       = ctx["situation"],
            recommendations = [
                EpisodeOut(
                    episode_title = r["metadata"]["episode_title"],
                    show_name     = r["metadata"]["show_name"],
                    url           = r["metadata"]["url"],
                    duration_mins = r["metadata"]["duration_mins"],
                    similarity    = r["similarity"],
                    explanation   = r["explanation"],
                )
                for r in recs
            ],
            memory_turn_count = len(state["memory"]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.delete("/api/memory/clear")
def clear_memory():
    """Reset conversation memory — start a fresh session."""
    state["memory"].clear()
    return {"status": "memory cleared", "turns": 0}
