import requests
from datetime import datetime
from app.config import settings

def fetch_posts():
    """Fetch posts from BlueSky based on hashtag"""
    if not settings.BLUESKY_USERNAME or not settings.BLUESKY_PASSWORD:
        raise ValueError("BlueSky credentials not found in environment variables")
    
    response = requests.post(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        json={"identifier": settings.BLUESKY_USERNAME, "password": settings.BLUESKY_PASSWORD}
    )
    response.raise_for_status()
    session_data = response.json()
    access_token = session_data["accessJwt"]

    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"q": settings.SEARCH_HASHTAG, "limit": settings.POST_LIMIT}

    response = requests.get(
        "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
        headers=headers,
        params=params
    )
    response.raise_for_status()
    posts = response.json().get("posts", [])

    print(f"[{datetime.now()}] Fetched {len(posts)} posts from BlueSky")
    for idx, post in enumerate(posts, 1):
        text = post["record"]["text"]
        print(f"{idx}. {text[:100]}...")
    
    return posts
