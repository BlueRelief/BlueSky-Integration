import google.generativeai as genai
from datetime import datetime
from app.config import settings
import json

def analyze_posts(posts):
    """Process posts with Gemini AI"""
    if not posts:
        print("No posts to process")
        return None
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    prompt = (
        "Analyze these social media posts about disasters and extract structured information. "
        "For each disaster mentioned, return a JSON array with objects containing:\n"
        "- location: specific location of the disaster\n"
        "- event_time: when the disaster occurred\n"
        "- severity: rate 1-5 (1=minor, 5=catastrophic)\n"
        "- magnitude: numerical magnitude if applicable (earthquake magnitude, hurricane category, etc.)\n"
        "- description: brief summary of the disaster\n\n"
        "Posts:\n"
        + "\n".join([f"{idx}. {post['record']['text']}" for idx, post in enumerate(posts, 1)])
        + "\n\nReturn ONLY valid JSON array format, no other text."
    )

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)
    analysis = response.text
    
    print(f"\n[{datetime.now()}] AI Analysis Complete:")
    print(analysis)
    
    return analysis