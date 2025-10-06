from fastapi import FastAPI
from app.routes import api
from app.database import init_db

app = FastAPI(
    title="BlueSky Disaster Monitoring",
    description="Automated data collection from BlueSky with AI-powered disaster analysis",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    """Initialize database on startup"""
    init_db()

app.include_router(api.router, prefix="/api", tags=["api"])

@app.get("/")
def root():
    return {
        "service": "BlueSky Disaster Monitoring",
        "version": "2.0.0",
        "docs": "/docs"
    }
