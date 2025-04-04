# Simple AI Agent

A FastAPI-based AI agent system that combines authentication, RAG (Retrieval Augmented Generation), and multiple specialized agents for handling different types of requests.

## Features

- **RAG System**: PDF document processing and semantic search using FAISS
- **Specialized Agents**:
  - Resume Agent: Answers questions about resumes using RAG
  - Email Agent: Sends emails using SendGrid
  - Weather Agent: Retrieves weather data from Open-Meteo API

## Tech Stack

- FastAPI
- SQLAlchemy
- OpenAI GPT-4
- FAISS Vector Store
- Cohere for reranking
- SendGrid for email
- Docker support

## Getting Started

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn main:app --reload
```

Or using Docker:
```bash
docker-compose up --build
```

## API Endpoints

- `POST /ai-agent`: Process general AI requests
- `GET /create-embeddings`: Generate embeddings for PDF documents

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `COHERE_API_KEY`: Your Cohere API key
- `SENDGRID_API_KEY`: SendGrid API key for email functionality

## Project Structure

```
├── agents/             # Specialized AI agents
├── auth/              # Authentication related code
├── config/            # Configuration files
├── content/           # PDF documents for RAG
├── faiss_index/       # FAISS vector store
├── models/            # SQLAlchemy models
├── repository/        # Database operations
├── routers/           # FastAPI routes
├── schemas/           # Pydantic models
├── services/          # Business logic
└── utils/             # Utility functions
```

## License

MIT