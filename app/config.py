import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BLUESKY_USERNAME = os.getenv("BlueSky_Username")
    BLUESKY_PASSWORD = os.getenv("BlueSky_Password")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_HASHTAG = os.getenv("SEARCH_HASHTAG", "#earthquake")
    POST_LIMIT = int(os.getenv("POST_LIMIT", "50"))
    SCHEDULE_HOURS = int(os.getenv("SCHEDULE_HOURS", "24"))
    
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bluesky:bluesky123@postgres:5432/bluesky")
    
    DATA_DIR = "data"

settings = Settings()
