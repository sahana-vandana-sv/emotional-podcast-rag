# API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Response:

```json
{
  "status": "healthy",
  "episodes_loaded": 20
}
```

### 2. Search Podcasts

**POST** `/api/search`

Request Body:

```json
{
  "query": "string (required)",
  "num_recommendations": "integer (optional, default: 3)"
}
```

Response:

```json
{
  "query": "string",
  "emotional_context": {
    "primary_emotion": "string",
    "situation": "string",
    "underlying_needs": ["string"],
    "search_keywords": ["string"]
  },
  "recommendations": [
    {
      "episode_title": "string",
      "show_name": "string",
      "url": "string",
      "duration_mins": "number",
      "explanation": "string",
      "similarity": "number (0-1)"
    }
  ]
}
```

## Error Responses

### 500 Internal Server Error

```json
{
  "detail": "Error message"
}
```

## Rate Limits

Currently no rate limiting implemented.

## Authentication

Currently no authentication required.
