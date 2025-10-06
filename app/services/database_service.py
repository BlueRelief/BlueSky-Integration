from datetime import datetime
from sqlalchemy.orm import Session
from app.models import CollectionRun, Post, Disaster
from app.database import SessionLocal

def create_collection_run() -> CollectionRun:
    """Create a new collection run"""
    db = SessionLocal()
    try:
        run = CollectionRun(status="running")
        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    finally:
        db.close()

def complete_collection_run(run_id: int, posts_count: int, status: str = "completed", error: str = None):
    """Mark collection run as complete"""
    db = SessionLocal()
    try:
        run = db.query(CollectionRun).filter(CollectionRun.id == run_id).first()
        if run:
            run.completed_at = datetime.utcnow()
            run.status = status
            run.posts_collected = posts_count
            if error:
                run.error_message = error
            db.commit()
    finally:
        db.close()

def save_posts(posts_data: list, run_id: int) -> int:
    """Save posts to database with deduplication"""
    db = SessionLocal()
    saved_count = 0
    
    try:
        for post_data in posts_data:
            bluesky_id = post_data.get("uri", "")
            
            existing = db.query(Post).filter(Post.bluesky_id == bluesky_id).first()
            if existing:
                continue
            
            post = Post(
                bluesky_id=bluesky_id,
                author_handle=post_data.get("author", {}).get("handle", ""),
                text=post_data.get("record", {}).get("text", ""),
                created_at=datetime.fromisoformat(post_data.get("record", {}).get("createdAt", "").replace("Z", "+00:00")),
                raw_data=post_data,
                collection_run_id=run_id
            )
            db.add(post)
            saved_count += 1
        
        db.commit()
        return saved_count
    finally:
        db.close()

def save_analysis(analysis_text: str, run_id: int):
    """Parse and save disaster data from AI analysis"""
    db = SessionLocal()
    
    try:
        lines = analysis_text.split("\n")
        current_disaster = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith(("**Location:**", "Location:")):
                if current_disaster:
                    _save_disaster(db, current_disaster, run_id)
                    current_disaster = {}
                current_disaster["location"] = line.split(":", 1)[-1].strip()
            
            elif line.startswith(("**Time:**", "Time:")):
                current_disaster["event_time"] = line.split(":", 1)[-1].strip()
            
            elif line.startswith(("**Severity:**", "Severity:")):
                severity_text = line.split(":", 1)[-1].strip()
                try:
                    severity = int(severity_text.split()[0])
                    current_disaster["severity"] = severity
                except:
                    pass
                current_disaster["description"] = severity_text
            
            elif "Magnitude" in line or "magnitude" in line:
                try:
                    import re
                    mag_match = re.search(r"(\d+\.?\d*)", line)
                    if mag_match:
                        current_disaster["magnitude"] = float(mag_match.group(1))
                except:
                    pass
        
        if current_disaster:
            _save_disaster(db, current_disaster, run_id)
        
        db.commit()
    finally:
        db.close()

def _save_disaster(db: Session, disaster_data: dict, run_id: int):
    """Helper to save a single disaster"""
    disaster = Disaster(
        location=disaster_data.get("location"),
        event_time=disaster_data.get("event_time"),
        severity=disaster_data.get("severity"),
        magnitude=disaster_data.get("magnitude"),
        description=disaster_data.get("description"),
        collection_run_id=run_id
    )
    db.add(disaster)

def get_recent_disasters(limit: int = 50):
    """Get recent disasters"""
    db = SessionLocal()
    try:
        disasters = db.query(Disaster).order_by(Disaster.extracted_at.desc()).limit(limit).all()
        return [
            {
                "id": d.id,
                "location": d.location,
                "event_time": d.event_time,
                "severity": d.severity,
                "magnitude": d.magnitude,
                "description": d.description,
                "extracted_at": d.extracted_at.isoformat()
            }
            for d in disasters
        ]
    finally:
        db.close()

def get_collection_stats():
    """Get collection statistics"""
    db = SessionLocal()
    try:
        total_runs = db.query(CollectionRun).count()
        total_posts = db.query(Post).count()
        total_disasters = db.query(Disaster).count()
        
        recent_runs = db.query(CollectionRun).order_by(CollectionRun.started_at.desc()).limit(5).all()
        
        return {
            "total_runs": total_runs,
            "total_posts": total_posts,
            "total_disasters": total_disasters,
            "recent_runs": [
                {
                    "id": r.id,
                    "started_at": r.started_at.isoformat(),
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "status": r.status,
                    "posts_collected": r.posts_collected
                }
                for r in recent_runs
            ]
        }
    finally:
        db.close()
