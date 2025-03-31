from fastapi import FastAPI
from routers import authentication, user, ai_agent, embedding_router
import models.models as models
from config.database import engine

app = FastAPI()

models.Base.metadata.create_all(engine)

app.include_router(authentication.router)
app.include_router(user.router)
app.include_router(ai_agent.router)
app.include_router(embedding_router.router)
