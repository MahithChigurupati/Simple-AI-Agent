from fastapi import FastAPI
from routers import ai_agent, embedding_router

app = FastAPI()

app.include_router(ai_agent.router)
app.include_router(embedding_router.router)
