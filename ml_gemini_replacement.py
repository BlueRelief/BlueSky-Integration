import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import json
import time
import psutil
import os
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

class MLGeminiReplacement:
    def __init__(self):
        self.disaster_model = None
        self.vectorizer = None
        self.nlp = None
        self.trained = False
        
    def load_spacy_model(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            return False
    
    def create_training_data(self):
        np.random.seed(42)
        
        disaster_posts = [
            "Earthquake M4.1 | Southwest of Sumatra, Indonesia | 6m ago",
            "Hurricane approaching Florida coast, evacuation ordered for Category 3 storm",
            "Tornado warning issued for central region, take shelter immediately",
            "Flooding reported in downtown area, roads closed due to heavy rainfall",
            "Wildfire spreading rapidly in forest area, 5000 acres burned",
            "Tsunami warning issued after magnitude 7.2 earthquake",
            "Avalanche blocks mountain pass, rescue teams deployed",
            "Volcanic eruption detected, ash cloud spreading",
            "Drought conditions worsen, water restrictions implemented",
            "Heat wave continues, temperatures reach record highs",
            "Blizzard conditions expected, travel not recommended",
            "Landslide blocks highway, emergency response activated",
            "Cyclone approaching coastal areas, prepare for impact",
            "Flash flood warning issued for low-lying areas",
            "Severe thunderstorm with hail and strong winds",
            "Ice storm causes power outages across region",
            "Dust storm reduces visibility to near zero",
            "River overflow threatens nearby communities",
            "Forest fire spreads to residential areas",
            "Earthquake swarm detected in fault zone"
        ]
        
        non_disaster_posts = [
            "Beautiful sunset today at the beach",
            "Coffee shop recommendations in downtown",
            "Local sports team wins championship",
            "New restaurant opening this weekend",
            "Traffic jam on highway 101",
            "Weather is nice today",
            "Going to the movies tonight",
            "Shopping at the mall",
            "Walking my dog in the park",
            "Reading a good book",
            "Cooking dinner for family",
            "Watching Netflix series",
            "Playing video games",
            "Listening to music",
            "Working from home today",
            "Meeting friends for lunch",
            "Cleaning the house",
            "Gardening in the backyard",
            "Taking a nap",
            "Planning vacation trip"
        ]
        
        all_posts = disaster_posts + non_disaster_posts
        labels = [1] * len(disaster_posts) + [0] * len(non_disaster_posts)
        
        for _ in range(2000):
            if np.random.random() > 0.5:
                disaster_keywords = ["earthquake", "hurricane", "flood", "fire", "storm", "tornado", "tsunami", 
                                   "avalanche", "volcanic", "drought", "heat wave", "blizzard", "landslide", 
                                   "cyclone", "flash flood", "thunderstorm", "ice storm", "dust storm", 
                                   "wildfire", "emergency", "disaster", "evacuation", "warning", "alert"]
                keyword = np.random.choice(disaster_keywords)
                
                templates = [
                    f"Emergency alert: {keyword} detected in area, please take precautions",
                    f"Breaking: {keyword} reported in the region, stay safe",
                    f"Warning: {keyword} approaching, evacuation may be necessary",
                    f"Alert: {keyword} detected, emergency services responding",
                    f"Urgent: {keyword} in progress, avoid affected areas",
                    f"Emergency: {keyword} warning issued for local area",
                    f"Breaking news: {keyword} causes damage in the area",
                    f"Alert: {keyword} emergency declared, take shelter",
                    f"OMG {keyword} is happening right now! Stay safe everyone",
                    f"Just heard about {keyword} on the news, hope everyone is okay",
                    f"Scary {keyword} situation developing, thoughts with those affected",
                    f"Praying for everyone affected by {keyword}",
                    f"Stay safe everyone, {keyword} is no joke",
                    f"Hope everyone is prepared for {keyword}",
                    f"Thinking of those dealing with {keyword} right now"
                ]
                post = np.random.choice(templates)
                
                if np.random.random() < 0.4:
                    noise_options = [
                        f"{post} #staySafe",
                        f"{post} omg",
                        f"{post} this is crazy",
                        f"{post} pls be careful",
                        f"{post} 🙏",
                        f"{post} #emergency",
                        f"{post} wtf",
                        f"{post} this is insane",
                        f"{post} #disaster",
                        f"{post} #breaking",
                        f"{post} #alert",
                        f"{post} #warning",
                        f"{post} #emergency",
                        f"{post} #staySafe",
                        f"{post} #prayers",
                        f"{post} #thoughts",
                        f"{post} #help",
                        f"{post} #rescue",
                        f"{post} #evacuation",
                        f"{post} #safety"
                    ]
                    post = np.random.choice(noise_options)
                
                all_posts.append(post)
                labels.append(1)
            else:
                normal_topics = ["weather", "food", "travel", "entertainment", "sports", "work", "music", 
                               "movies", "books", "technology", "politics", "education", "health", "fashion",
                               "art", "science", "business", "family", "friends", "hobbies"]
                topic = np.random.choice(normal_topics)
                
                templates = [
                    f"Just enjoying {topic} today, hope everyone has a great day",
                    f"Having a wonderful time with {topic}, life is good",
                    f"Excited about {topic}, can't wait to share more",
                    f"Love spending time on {topic}, it's so relaxing",
                    f"Great day for {topic}, feeling grateful",
                    f"Just finished working on {topic}, feeling accomplished",
                    f"Amazing {topic} experience today, highly recommend",
                    f"Thinking about {topic}, always brings a smile",
                    f"Just discovered something cool about {topic}",
                    f"Perfect day for {topic}, everything is going well",
                    f"What a beautiful day for {topic}",
                    f"Can't get enough of {topic} lately",
                    f"Anyone else love {topic} as much as I do?",
                    f"Just had the best {topic} experience",
                    f"Looking forward to more {topic} this weekend"
                ]
                post = np.random.choice(templates)
                
                if np.random.random() < 0.5:
                    noise_options = [
                        f"{post} #blessed",
                        f"{post} lol",
                        f"{post} this is amazing",
                        f"{post} can't believe it",
                        f"{post} 😍",
                        f"{post} #life",
                        f"{post} omg",
                        f"{post} this is crazy",
                        f"{post} wtf",
                        f"{post} this is insane",
                        f"{post} #goodvibes",
                        f"{post} #happy",
                        f"{post} #love",
                        f"{post} #blessed",
                        f"{post} #grateful",
                        f"{post} #thankful",
                        f"{post} #peace",
                        f"{post} #joy",
                        f"{post} #smile",
                        f"{post} #positive",
                        f"{post} #mindfulness",
                        f"{post} #wellness",
                        f"{post} #selfcare",
                        f"{post} #motivation",
                        f"{post} #inspiration"
                    ]
                    post = np.random.choice(noise_options)
                
                all_posts.append(post)
                labels.append(0)
        
        ambiguous_posts = [
            "Storm coming this weekend, hope it's not too bad",
            "Heavy rain expected, roads might flood",
            "Wind picking up, might be a problem",
            "Thunderstorm warning issued for tonight",
            "Flash flood watch in effect",
            
            "Fire drill at work today, everyone was confused",
            "Emergency meeting called, not sure what's happening",
            "Test alert: this is just a drill",
            "Emergency broadcast system test",
            "Fire alarm went off, false alarm",
            
            "Warning: this movie is a disaster",
            "Breaking: my phone just died",
            "Alert: traffic is terrible today",
            "Emergency: I need coffee",
            "Disaster averted, found my keys",
            "Warning: this restaurant is amazing",
            "Breaking news: I just woke up",
            "Emergency: my cat is being dramatic",
            "Alert: it's raining outside",
            "Disaster: forgot my umbrella",
            "Warning: this book is addictive",
            "Breaking: just finished my homework",
            "Emergency: need to buy groceries",
            "Alert: my plants need water",
            "Disaster: my internet is slow",
            "Warning: this game is too hard",
            "Breaking: just saw a shooting star",
            
            "Power outage in my neighborhood",
            "Road closed due to construction",
            "Building evacuated for maintenance",
            "Water main break on Main Street",
            "Gas leak reported in downtown",
            "Bridge closed for inspection",
            "Tunnel blocked by debris",
            "Highway closed due to accident",
            "Train service suspended",
            "Airport delays due to weather",
            
            "Fire at the old factory",
            "Explosion heard downtown",
            "Smoke visible from the hills",
            "Loud noise woke everyone up",
            "Police blocking the street",
            "Ambulance and fire trucks everywhere",
            "Helicopter circling overhead",
            "Emergency vehicles on the scene",
            "Road blocked by authorities",
            "Evacuation order issued",
            
            "emrgency alert: earthquak in area",
            "warnign: flood risk high",
            "breking: storm aproaching",
            "alert: fire danger extreme",
            "urgent: tornado warning",
            "disaster: hurricane path uncertain",
            "emergency: tsunami threat",
            "warning: avalanche risk",
            "alert: volcanic activity",
            "breaking: landslide reported"
        ]
        
        for post in ambiguous_posts:
            all_posts.append(post)
            if any(word in post.lower() for word in ["drill", "test", "false", "movie", "phone", "coffee", "keys", "restaurant", "woke", "cat", "umbrella", "book", "homework", "groceries", "plants", "internet", "game", "shooting star"]):
                labels.append(0)
            elif any(word in post.lower() for word in ["power outage", "gas leak", "water main", "bridge closed", "tunnel blocked", "highway closed", "train service", "airport delays"]):
                labels.append(1)
            elif any(word in post.lower() for word in ["fire", "explosion", "smoke", "loud noise", "police", "ambulance", "fire trucks", "helicopter", "evacuation"]):
                labels.append(1)
            else:
                labels.append(0)
        
        # Add realistic edge cases and ambiguous posts
        edge_case_posts = [
            # Sarcastic/joke posts that mention disasters
            "My code is a disaster today 😅",
            "This traffic is catastrophic",
            "My phone battery died - it's an emergency!",
            "Breaking: I just woke up",
            "Alert: my coffee is cold",
            "Emergency: I need more sleep",
            "Disaster averted: found my keys",
            "Warning: this movie is terrible",
            "Breaking news: I'm hungry",
            "Emergency: my internet is slow",
            "Catastrophic: forgot my password",
            "Devastating: my favorite show ended",
            "Massive: this pizza is huge",
            "Severe: I'm tired",
            "Urgent: need to pee",
            
            # Metaphorical uses of disaster words
            "This meeting is a disaster",
            "My presentation was catastrophic",
            "The project is in crisis",
            "This situation is critical",
            "We're in emergency mode at work",
            "This is a major disaster for our team",
            "The deadline is approaching like a storm",
            "This bug is causing chaos",
            "The system is in crisis",
            "This is a critical issue",
            
            # News/entertainment posts
            "Watching disaster movies tonight",
            "Reading about historical disasters",
            "Learning about natural disasters in school",
            "Documentary about disasters was interesting",
            "Disaster preparedness tips from the news",
            "Emergency services training program",
            "Disaster relief fund donation",
            "Emergency response team training",
            "Disaster management course",
            "Emergency preparedness workshop",
            
            # Weather that's not disasters
            "Nice weather today, no storms",
            "Beautiful sunny day, no disasters here",
            "Light rain, nothing serious",
            "Breeze is nice, no strong winds",
            "Cloudy but peaceful",
            "Mild weather, no extremes",
            "Calm day, no emergencies",
            "Quiet weather, all good",
            "Pleasant day, no alerts",
            "Peaceful weather, no warnings",
            
            # False alarms and drills
            "Fire drill at school today",
            "Emergency drill at work",
            "Test alert - this is just a drill",
            "Emergency broadcast system test",
            "Fire alarm test in building",
            "Emergency evacuation drill",
            "Disaster preparedness exercise",
            "Emergency response drill",
            "Crisis management training",
            "Emergency simulation exercise",
            
            # Context-dependent posts
            "Storm coming this weekend, hope it's not too bad",
            "Heavy rain expected, roads might flood",
            "Wind picking up, might be a problem",
            "Thunderstorm warning issued for tonight",
            "Flash flood watch in effect",
            "High wind warning for tomorrow",
            "Severe weather possible this week",
            "Storm system approaching the area",
            "Weather alert issued for tonight",
            "Storm warning for coastal areas",
            
            # Ambiguous emergency situations
            "Power outage in my neighborhood",
            "Road closed due to construction",
            "Building evacuated for maintenance",
            "Water main break on Main Street",
            "Gas leak reported in downtown",
            "Bridge closed for inspection",
            "Tunnel blocked by debris",
            "Highway closed due to accident",
            "Train service suspended",
            "Airport delays due to weather",
            
            # Social media drama
            "This is a disaster for my reputation",
            "My relationship is in crisis",
            "This situation is critical for my career",
            "We're in emergency mode with this project",
            "This is a major disaster for our business",
            "The deadline is approaching like a storm",
            "This bug is causing chaos in production",
            "The system is in crisis mode",
            "This is a critical issue for our team",
            "We're facing a disaster with this client",
            
            # Gaming/entertainment
            "Disaster in the game, lost all my progress",
            "Emergency: need to save my game",
            "Critical bug in the software",
            "System crash was catastrophic",
            "Emergency restart required",
            "Disaster recovery in progress",
            "Critical error in the application",
            "Emergency patch needed",
            "System failure was devastating",
            "Critical update required",
            
            # Sports/competition
            "Disaster of a game for our team",
            "Catastrophic loss in the championship",
            "Critical moment in the match",
            "Emergency timeout called",
            "Disaster for our season",
            "Critical play in the game",
            "Emergency substitution made",
            "Disaster for our playoff chances",
            "Critical error by the player",
            "Emergency situation in the game"
        ]
        
        for post in edge_case_posts:
            all_posts.append(post)
            # Most of these should be classified as non-disasters
            if any(word in post.lower() for word in ["code", "traffic", "phone", "woke", "coffee", "keys", "movie", "hungry", "internet", "password", "show", "pizza", "tired", "pee", "meeting", "presentation", "project", "deadline", "bug", "system", "watching", "reading", "learning", "documentary", "nice", "beautiful", "light", "breeze", "cloudy", "mild", "calm", "quiet", "pleasant", "peaceful", "drill", "test", "exercise", "training", "simulation", "weekend", "might", "possible", "construction", "maintenance", "inspection", "accident", "reputation", "relationship", "career", "business", "client", "game", "software", "application", "team", "season", "championship", "match", "playoff", "player"]):
                labels.append(0)
            else:
                labels.append(1)
        
        # Add more realistic social media noise and variations
        additional_noise_posts = []
        additional_noise_labels = []
        
        for i in range(800):
            if np.random.random() > 0.3:
                # More disaster posts with realistic social media noise
                disaster_types = ["earthquake", "flood", "fire", "storm", "tornado", "hurricane", "wildfire", "tsunami"]
                disaster = np.random.choice(disaster_types)
                
                locations = ["California", "Texas", "Florida", "New York", "Washington", "Oregon", "Arizona", "Nevada", "Utah", "Colorado"]
                location = np.random.choice(locations)
                
                social_media_templates = [
                    f"OMG {disaster} in {location} right now! Stay safe everyone 🙏",
                    f"Breaking: {disaster} reported in {location}. Thoughts with everyone affected",
                    f"Just heard about {disaster} in {location}. Hope everyone is okay",
                    f"Scary {disaster} situation in {location}. Please stay safe",
                    f"Praying for everyone in {location} dealing with {disaster}",
                    f"Emergency alert: {disaster} in {location}. Take precautions",
                    f"Devastating {disaster} in {location}. My heart goes out to everyone",
                    f"Urgent: {disaster} warning for {location}. Evacuate if necessary",
                    f"Terrible {disaster} in {location}. Stay strong everyone",
                    f"Emergency services responding to {disaster} in {location}",
                    f"Massive {disaster} in {location}. Please share this",
                    f"Breaking news: {disaster} causes damage in {location}",
                    f"Alert: {disaster} emergency in {location}. Stay indoors",
                    f"Catastrophic {disaster} in {location}. Help needed",
                    f"Emergency broadcast: {disaster} in {location}. Take shelter",
                    f"OMG {disaster} in {location} right now! Stay safe everyone 🙏",
                    f"Breaking: {disaster} reported in {location}. Thoughts with everyone affected",
                    f"Just heard about {disaster} in {location}. Hope everyone is okay",
                    f"Scary {disaster} situation in {location}. Please stay safe",
                    f"Praying for everyone in {location} dealing with {disaster}",
                    f"Emergency alert: {disaster} in {location}. Take precautions",
                    f"Devastating {disaster} in {location}. My heart goes out to everyone",
                    f"Urgent: {disaster} warning for {location}. Evacuate if necessary",
                    f"Terrible {disaster} in {location}. Stay strong everyone",
                    f"Emergency services responding to {disaster} in {location}",
                    f"Massive {disaster} in {location}. Please share this",
                    f"Breaking news: {disaster} causes damage in {location}",
                    f"Alert: {disaster} emergency in {location}. Stay indoors",
                    f"Catastrophic {disaster} in {location}. Help needed",
                    f"Emergency broadcast: {disaster} in {location}. Take shelter"
                ]
                
                post = np.random.choice(social_media_templates)
                
                # Add more realistic noise
                if np.random.random() < 0.5:
                    noise_additions = [
                        " #emergency", " #disaster", " #staySafe", " #prayers", " #thoughts", 
                        " #help", " #rescue", " #evacuation", " #safety", " #breaking",
                        " omg", " this is crazy", " wtf", " this is insane", " pls be careful",
                        " 🙏", " 😢", " 😰", " 😱", " 💔",
                        " #emergency", " #disaster", " #staySafe", " #prayers", " #thoughts", 
                        " #help", " #rescue", " #evacuation", " #safety", " #breaking",
                        " omg", " this is crazy", " wtf", " this is insane", " pls be careful",
                        " 🙏", " 😢", " 😰", " 😱", " 💔"
                    ]
                    post += np.random.choice(noise_additions)
                
                additional_noise_posts.append(post)
                additional_noise_labels.append(1)
            else:
                # More normal posts with realistic social media noise
                normal_topics = ["coffee", "work", "gym", "food", "music", "movies", "books", "travel", "family", "friends", "weather", "shopping", "cooking", "reading", "walking", "driving", "studying", "sleeping", "eating", "drinking"]
                topic = np.random.choice(normal_topics)
                
                social_media_templates = [
                    f"Just enjoying {topic} today. Life is good!",
                    f"Having a great time with {topic}. Feeling blessed",
                    f"Love spending time on {topic}. So relaxing",
                    f"Perfect day for {topic}. Everything is going well",
                    f"Can't get enough of {topic} lately. Highly recommend",
                    f"Amazing {topic} experience today. Feeling grateful",
                    f"Just finished {topic}. Feeling accomplished",
                    f"Wonderful {topic} session. Mind is clear",
                    f"Great {topic} day. Feeling positive",
                    f"Enjoying {topic} right now. Life is beautiful",
                    f"Just discovered something cool about {topic}",
                    f"Thinking about {topic}. Always brings a smile",
                    f"Perfect moment for {topic}. Feeling peaceful",
                    f"Just had the best {topic} experience",
                    f"Looking forward to more {topic} this weekend",
                    f"Just enjoying {topic} today. Life is good!",
                    f"Having a great time with {topic}. Feeling blessed",
                    f"Love spending time on {topic}. So relaxing",
                    f"Perfect day for {topic}. Everything is going well",
                    f"Can't get enough of {topic} lately. Highly recommend",
                    f"Amazing {topic} experience today. Feeling grateful",
                    f"Just finished {topic}. Feeling accomplished",
                    f"Wonderful {topic} session. Mind is clear",
                    f"Great {topic} day. Feeling positive",
                    f"Enjoying {topic} right now. Life is beautiful",
                    f"Just discovered something cool about {topic}",
                    f"Thinking about {topic}. Always brings a smile",
                    f"Perfect moment for {topic}. Feeling peaceful",
                    f"Just had the best {topic} experience",
                    f"Looking forward to more {topic} this weekend"
                ]
                
                post = np.random.choice(social_media_templates)
                
                # Add more realistic noise
                if np.random.random() < 0.6:
                    noise_additions = [
                        " #blessed", " #grateful", " #happy", " #love", " #life", " #goodvibes",
                        " #peace", " #joy", " #smile", " #positive", " #mindfulness", " #wellness",
                        " lol", " this is amazing", " can't believe it", " omg", " this is crazy",
                        " 😍", " 😊", " 😄", " 😁", " 🤗", " 💕", " ✨", " 🌟",
                        " #blessed", " #grateful", " #happy", " #love", " #life", " #goodvibes",
                        " #peace", " #joy", " #smile", " #positive", " #mindfulness", " #wellness",
                        " lol", " this is amazing", " can't believe it", " omg", " this is crazy",
                        " 😍", " 😊", " 😄", " 😁", " 🤗", " 💕", " ✨", " 🌟"
                    ]
                    post += np.random.choice(noise_additions)
                
                additional_noise_posts.append(post)
                additional_noise_labels.append(0)
        
        all_posts.extend(additional_noise_posts)
        labels.extend(additional_noise_labels)
        
        noisy_posts = []
        noisy_labels = []
        
        for i, post in enumerate(all_posts):
            # Add realistic typos and misspellings
            if np.random.random() < 0.50:
                post = post.replace("emergency", "emrgency")
                post = post.replace("warning", "warnign")
                post = post.replace("breaking", "breking")
                post = post.replace("alert", "alret")
                post = post.replace("disaster", "disater")
                post = post.replace("earthquake", "earthqake")
                post = post.replace("hurricane", "huricane")
                post = post.replace("flood", "flod")
                post = post.replace("fire", "fier")
                post = post.replace("storm", "stom")
                post = post.replace("evacuation", "evacution")
                post = post.replace("emergency", "emrgency")
                post = post.replace("situation", "situaton")
                post = post.replace("devastating", "devastating")
                post = post.replace("catastrophic", "catastrophic")
                post = post.replace("massive", "masive")
                post = post.replace("terrible", "terible")
                post = post.replace("urgent", "urgent")
                post = post.replace("urgent", "urgent")
                post = post.replace("severe", "severe")
                post = post.replace("critical", "critical")
                post = post.replace("dangerous", "dangerous")
                post = post.replace("threat", "threat")
                post = post.replace("crisis", "crisis")
                post = post.replace("chaos", "chaos")
                post = post.replace("destruction", "destruction")
                post = post.replace("damage", "damage")
                post = post.replace("injury", "injury")
                post = post.replace("casualty", "casualty")
                post = post.replace("victim", "victim")
                post = post.replace("rescue", "rescue")
                post = post.replace("evacuate", "evacuate")
                post = post.replace("shelter", "shelter")
                post = post.replace("warning", "warnign")
                post = post.replace("alert", "alret")
                post = post.replace("breaking", "breking")
                post = post.replace("emergency", "emrgency")
                post = post.replace("disaster", "disater")
                post = post.replace("earthquake", "earthqake")
                post = post.replace("hurricane", "huricane")
                post = post.replace("flood", "flod")
                post = post.replace("fire", "fier")
                post = post.replace("storm", "stom")
                post = post.replace("tornado", "tornado")
                post = post.replace("tsunami", "tsunami")
                post = post.replace("wildfire", "wildfire")
                post = post.replace("avalanche", "avalanche")
                post = post.replace("volcanic", "volcanic")
                post = post.replace("drought", "drought")
                post = post.replace("blizzard", "blizzard")
                post = post.replace("landslide", "landslide")
                post = post.replace("cyclone", "cyclone")
                post = post.replace("thunderstorm", "thunderstorm")
                post = post.replace("ice storm", "ice storm")
                post = post.replace("dust storm", "dust storm")
            
            # Add random characters (like autocorrect errors)
            if np.random.random() < 0.25:
                random_chars = [" xyz", " abc", " 123", " qwe", " asd", " zxc", " lol", " omg", " wtf", " btw", " fyi", " tbh", " imo", " ngl", " fr", " no cap", " periodt", " slay", " queen", " king", " vibe", " mood", " energy", " aesthetic", " iconic", " legendary", " fire", " lit", " savage", " flex", " clout", " drip", " bussin", " no cap", " periodt", " slay", " queen", " king", " vibe", " mood", " energy", " aesthetic", " iconic", " legendary", " fire", " lit", " savage", " flex", " clout", " drip", " bussin"]
                post = post + np.random.choice(random_chars)
            
            # Add extra spaces (like mobile typing errors)
            if np.random.random() < 0.30:
                post = post.replace(" ", "  ")
            
            # Add realistic social media abbreviations
            if np.random.random() < 0.50:
                post = post.replace("you", "u")
                post = post.replace("your", "ur")
                post = post.replace("are", "r")
                post = post.replace("be", "b")
                post = post.replace("to", "2")
                post = post.replace("for", "4")
                post = post.replace("and", "&")
                post = post.replace("with", "w/")
                post = post.replace("because", "bc")
                post = post.replace("please", "pls")
                post = post.replace("thanks", "thx")
                post = post.replace("okay", "ok")
                post = post.replace("right", "rite")
                post = post.replace("tonight", "2nite")
                post = post.replace("tomorrow", "2moro")
                post = post.replace("today", "2day")
                post = post.replace("tonight", "2nite")
                post = post.replace("tomorrow", "2moro")
                post = post.replace("today", "2day")
                post = post.replace("emergency", "emrgency")
                post = post.replace("warning", "warnign")
                post = post.replace("breaking", "breking")
                post = post.replace("alert", "alret")
                post = post.replace("disaster", "disater")
                post = post.replace("earthquake", "earthqake")
                post = post.replace("hurricane", "huricane")
                post = post.replace("flood", "flod")
                post = post.replace("fire", "fier")
                post = post.replace("storm", "stom")
                post = post.replace("evacuation", "evacution")
                post = post.replace("situation", "situaton")
                post = post.replace("devastating", "devastating")
                post = post.replace("catastrophic", "catastrophic")
                post = post.replace("massive", "masive")
                post = post.replace("terrible", "terible")
                post = post.replace("urgent", "urgent")
                post = post.replace("severe", "severe")
                post = post.replace("critical", "critical")
                post = post.replace("dangerous", "dangerous")
                post = post.replace("threat", "threat")
                post = post.replace("crisis", "crisis")
                post = post.replace("chaos", "chaos")
                post = post.replace("destruction", "destruction")
                post = post.replace("damage", "damage")
                post = post.replace("injury", "injury")
                post = post.replace("casualty", "casualty")
                post = post.replace("victim", "victim")
                post = post.replace("rescue", "rescue")
                post = post.replace("evacuate", "evacuate")
                post = post.replace("shelter", "shelter")
                post = post.replace("warning", "warnign")
                post = post.replace("alert", "alret")
                post = post.replace("breaking", "breking")
                post = post.replace("emergency", "emrgency")
                post = post.replace("disaster", "disater")
                post = post.replace("earthquake", "earthqake")
                post = post.replace("hurricane", "huricane")
                post = post.replace("flood", "flod")
                post = post.replace("fire", "fier")
                post = post.replace("storm", "stom")
                post = post.replace("tornado", "tornado")
                post = post.replace("tsunami", "tsunami")
                post = post.replace("wildfire", "wildfire")
                post = post.replace("avalanche", "avalanche")
                post = post.replace("volcanic", "volcanic")
                post = post.replace("drought", "drought")
                post = post.replace("blizzard", "blizzard")
                post = post.replace("landslide", "landslide")
                post = post.replace("cyclone", "cyclone")
                post = post.replace("thunderstorm", "thunderstorm")
                post = post.replace("ice storm", "ice storm")
                post = post.replace("dust storm", "dust storm")
            
            noisy_posts.append(post)
            noisy_labels.append(labels[i])
        
        all_posts = noisy_posts
        labels = noisy_labels
        
        return pd.DataFrame({
            'text': all_posts,
            'label': labels
        })
    
    def train(self):
        print("Training ML + NER Pipeline...")
        print("=" * 50)
        
        if not self.load_spacy_model():
            return False
        
        df = self.create_training_data()
        X = df['text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.disaster_model = LogisticRegression(random_state=42, max_iter=1000)
        self.disaster_model.fit(X_train_vec, y_train)
        
        y_pred = self.disaster_model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Disaster Detection Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        self.trained = True
        return True
    
    def extract_location(self, text):
        if not self.nlp:
            return "Unknown location"
        
        # Convert to string to handle numpy types
        text = str(text)
        doc = self.nlp(text)
        locations = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                locations.append(ent.text)
        
        return locations[0] if locations else "Unknown location"
    
    def extract_time(self, text):
        # Convert to string to handle numpy types
        text = str(text)
        time_patterns = [
            r'(\d+[mh]?\s*ago)',
            r'(\d+:\d+\s*(?:am|pm)?)',
            r'(today|yesterday|tomorrow)',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Unknown time"
    
    def predict_severity(self, text):
        # Convert to string to handle numpy types
        text = str(text)
        severity_keywords = {
            5: ["catastrophic", "devastating", "massive", "severe", "emergency", "evacuation"],
            4: ["major", "significant", "serious", "dangerous", "warning"],
            3: ["moderate", "considerable", "substantial", "alert"],
            2: ["minor", "small", "light", "brief"],
            1: ["slight", "minimal", "low", "weak"]
        }
        
        text_lower = text.lower()
        for level, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return 3
    
    def extract_magnitude(self, text):
        # Convert to string to handle numpy types
        text = str(text)
        magnitude_patterns = [
            r'M(\d+\.?\d*)',
            r'Category\s*(\d+)',
            r'(\d+\.?\d*)\s*mph',
            r'(\d+\.?\d*)\s*inches',
            r'(\d+\.?\d*)\s*feet',
        ]
        
        for pattern in magnitude_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def analyze_post(self, post_text):
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        start_time = time.time()
        
        text_vec = self.vectorizer.transform([post_text])
        is_disaster = self.disaster_model.predict(text_vec)[0]
        
        if not is_disaster:
            return None
        
        location = self.extract_location(post_text)
        event_time = self.extract_time(post_text)
        severity = self.predict_severity(post_text)
        magnitude = self.extract_magnitude(post_text)
        description = post_text[:100] + "..." if len(post_text) > 100 else post_text
        
        processing_time = time.time() - start_time
        
        return {
            "location": location,
            "event_time": event_time,
            "severity": severity,
            "magnitude": magnitude,
            "description": description,
            "processing_time_ms": round(processing_time * 1000, 2)
        }
    
    def analyze_posts_batch(self, posts):
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        results = []
        for post in posts:
            result = self.analyze_post(post)
            if result:
                results.append(result)
        
        return results
    
    def benchmark_performance(self, num_posts=1000):
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"\nBenchmarking Performance on {num_posts} posts...")
        print("=" * 50)
        
        df = self.create_training_data()
        test_posts = df['text'].sample(num_posts, random_state=42).tolist()
        
        start_time = time.time()
        results = self.analyze_posts_batch(test_posts)
        total_time = time.time() - start_time
        
        disaster_posts = len(results)
        avg_time_per_post = total_time / num_posts * 1000
        
        print(f"Total posts processed: {num_posts}")
        print(f"Disaster posts detected: {disaster_posts}")
        print(f"Total processing time: {total_time:.3f}s")
        print(f"Average time per post: {avg_time_per_post:.2f}ms")
        print(f"Posts per second: {num_posts / total_time:.1f}")
        
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_usage:.1f} MB")
        
        return {
            "total_posts": num_posts,
            "disaster_posts": disaster_posts,
            "total_time": total_time,
            "avg_time_per_post_ms": avg_time_per_post,
            "posts_per_second": num_posts / total_time,
            "memory_usage_mb": memory_usage
        }

def main():
    print("ML + NER Pipeline - Gemini Replacement")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    pipeline = MLGeminiReplacement()
    
    if not pipeline.train():
        print("Failed to train pipeline")
        return
    
    print("\nTesting on Sample Posts:")
    print("=" * 50)
    
    test_posts = [
        "Earthquake M4.1 | Southwest of Sumatra, Indonesia | 6m ago",
        "Hurricane approaching Florida coast, evacuation ordered for Category 3 storm",
        "Tornado warning issued for central region, take shelter immediately",
        "Beautiful sunset today at the beach",
        "Fire drill at work today, everyone was confused",
        "Power outage in my neighborhood, emergency services responding",
        "Breaking: massive wildfire spreading in California, evacuation ordered",
        "Just had coffee with friends, great day!",
        "Tsunami warning issued after magnitude 7.2 earthquake in Pacific",
        "Traffic jam on highway 101, running late to work"
    ]
    
    for i, post in enumerate(test_posts, 1):
        print(f"\n{i}. Post: {post}")
        result = pipeline.analyze_post(post)
        if result:
            print(f"   Location: {result['location']}")
            print(f"   Event Time: {result['event_time']}")
            print(f"   Severity: {result['severity']}/5")
            print(f"   Magnitude: {result['magnitude']}")
            print(f"   Processing Time: {result['processing_time_ms']}ms")
        else:
            print("   Not classified as disaster")
    
    benchmark_results = pipeline.benchmark_performance(1000)
    
    print(f"\nPipeline ready for production use!")
    print(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
