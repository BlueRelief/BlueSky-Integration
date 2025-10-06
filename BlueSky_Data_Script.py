# BlueSky_Data_script.py

from fastapi import FastAPI
from atproto import Client
import os
from dotenv import load_dotenv
import google.generativeai as genai

app = FastAPI()

# -----------------------
# 1. Helper Functions
# -----------------------
def load_data():
    # Original logic from notebook
    username = os.getenv("BlueSky_Username")
    app_password = os.getenv("BlueSky_Password")
    client.login(username, app_password)
    client = Client()

    response = requests.post(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        json={"identifier": username, "password": app_password}
    )
    response.raise_for_status()
    session_data = response.json()
    access_token = session_data["accessJwt"]

    headers = {"Authorization": f"Bearer {access_token}"}

    hashtag = "#earthquake"  # include the `#`
    params = {
        "q": hashtag,
        "limit": 50
    }

    response = requests.get(
        "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
        headers=headers,
        params=params
    )
    response.raise_for_status()
    posts = response.json().get("posts", [])

    # Extract post text
    for idx, post in enumerate(posts, 1):
        text = post["record"]["text"]
        print(f"{idx}. {text}\n")
    pass

def process_data(data):
    prompt = (
        "Extract the location of the disaster, time, and rate the severity on a scale of 1-5:\n\n"
        + "\n".join([f"{idx}. {post['record']['text']}" for idx, post in enumerate(posts, 1)])
    )

    # Initialize Gemini model (make sure you have set your API key)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)
    print(response.text)
    pass

def visualize_results(results):
    # Original visualization logic
    pass

# -----------------------
# 2. Main Execution Pipeline
# -----------------------
def main():
    data = load_data()
    results = process_data(data)

# -----------------------
# 3. FastAPI Endpoint
# -----------------------
@app.get("/run-demo")
def run_demo():
    main()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
