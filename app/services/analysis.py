import google.generativeai as genai
from datetime import datetime
from app.config import settings

def analyze_posts(posts):
    """Process posts with Gemini AI"""
    if not posts:
        print("No posts to process")
        return None
    
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    prompt = (
        "Extract the location of the disaster, time, and rate the severity on a scale of 1-5:\n\n"
        + "\n".join([f"{idx}. {post['record']['text']}" for idx, post in enumerate(posts, 1)])
    )

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)
    analysis = response.text
    
    print(f"\n[{datetime.now()}] AI Analysis Complete:")
    print(analysis)
    
    return analysis