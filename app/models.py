from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class CollectionRun(Base):
    __tablename__ = "collection_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False, default="running")
    posts_collected = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    posts = relationship("Post", back_populates="collection_run")
    disasters = relationship("Disaster", back_populates="collection_run")

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    bluesky_id = Column(String(255), unique=True, nullable=False, index=True)
    author_handle = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    raw_data = Column(JSON, nullable=True)
    collection_run_id = Column(Integer, ForeignKey("collection_runs.id"), nullable=False)
    
    collection_run = relationship("CollectionRun", back_populates="posts")

class Disaster(Base):
    __tablename__ = "disasters"
    
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String(500), nullable=True)
    event_time = Column(String(255), nullable=True)
    severity = Column(Integer, nullable=True)
    magnitude = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    extracted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    collection_run_id = Column(Integer, ForeignKey("collection_runs.id"), nullable=False)
    
    collection_run = relationship("CollectionRun", back_populates="disasters")
