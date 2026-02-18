# 🎧 Emotional Podcast RAG System

AI powered podcast discovery system that understands your emotional state and recommends relevant episodes.

## 🌐 Live Demo
**API:** https://emotional-podcast-rag.onrender.com  
**Docs:** https://emotional-podcast-rag.onrender.com/docs  

Try it now — no code required! Open the docs page and click "Try it out" on `/api/search`.

[![Deploy Status](https://img.shields.io/badge/deploy-passing-brightgreen)](https://emotional-podcast-rag.onrender.com)
[![API Status](https://img.shields.io/badge/API-online-success)](https://emotional-podcast-rag.com/health)

---

## 🌟 What It Does

When you're stressed, anxious, or going through something hard — this system finds the right podcast episode for you...
---

## 🌟 Features

- **Emotional Intelligence**: Understands emotional context of your query
- **Semantic Search**: Finds relevant episodes based on semantic meaning, not just keywords
- **Personalized Explanations**: Tells you why each episode will help
- **Fast API**: RESTful API for easy integration

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/sahana-vandana-sv/emotional-podcast-rag.git
cd emotional-podcast-rag
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

4. Add your data
```bash
# Place your transcripts_with_embeddings.csv in:
# data/processed/transcripts_with_embeddings.csv
```

5. Run the API
```bash
cd api
uvicorn main:app --reload --port 8000
```

6. Visit http://localhost:8000/docs

## 📖 API Usage

### Search Endpoint
```bash
POST /api/search

{
  "query": "I feel anxious about starting a new job",
  "num_recommendations": 3
}
```

### Response
```json
{
  "query": "I feel anxious about starting a new job",
  "emotional_context": {
    "primary_emotion": "anxiety",
    "situation": "career transition",
    "underlying_needs": ["reassurance", "practical advice"]
  },
  "recommendations": [
    {
      "episode_title": "Managing Career Transitions",
      "show_name": "WorkLife",
      "url": "https://youtube.com/...",
      "duration_mins": 45,
      "explanation": "This episode addresses...",
      "similarity": 0.85
    }
  ]
}
```

## 🏗️ Architecture
```
emotional-podcast-rag/
├── src/               # Core Python modules
│   ├── search.py      # Semantic search
│   ├── llm.py         # LLM interactions
│   ├── rag_pipeline.py # Main pipeline
│   └── data_loader.py # Data utilities
├── api/               # FastAPI application
│   └── main.py
├── data/              # Data storage (not in git)
│   └── processed/
└── tests/             # Unit tests
```

## 🧪 Testing
```bash
python tests/test_search.py
```

## 🛠️ Tech Stack

- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4o-mini
- **Search**: Cosine similarity
- **API**: FastAPI
- **Data**: Pandas, NumPy

## 📊 Dataset

- 20 podcast episodes
- Topics: Mental health, personal development, emotional wellness
- ~200,000 words of content

## 🤝 Contributing

Contributions welcome! Please open an issue first.

## 📄 License

MIT License

## 👤 Author

Sahana Vandana - [GitHub](https://github.com/sahana-vandana-sv)

## 🙏 Acknowledgments

- OpenAI for embeddings and LLM
- All podcast creators whose content powers this system
