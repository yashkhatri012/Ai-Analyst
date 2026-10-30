
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.conversations import router as conversations_router
from app.db.database import close_database, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        yield
    finally:
        await close_database()


app = FastAPI(
    title="AI Analyst API",
    lifespan=lifespan,
)

app.include_router(conversations_router)
