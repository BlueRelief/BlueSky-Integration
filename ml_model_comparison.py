import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import time
import psutil
import os
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

def load_real_disaster_data():
    try:
        print("Loading real disaster tweets dataset...")
        from datasets import load_dataset
        
        dataset = load_dataset("tweet_eval", "disaster", split="train")
        df = dataset.to_pandas()
        
        df = df.dropna(subset=['text', 'label'])
        df = df[df['text'].str.len() > 10]
        df = df.rename(columns={'text': 'text', 'label': 'label'})
        df = df.sample(n=min(2000, len(df)), random_state=42)
        
        print(f"Loaded {len(df)} real disaster tweets")
        print(f"Disaster tweets: {sum(df['label'])}")
        print(f"Non-disaster tweets: {len(df) - sum(df['label'])}")
        
        return df
        
    except Exception as e:
        print(f"Failed to load real dataset: {e}")
        print("Falling back to synthetic data...")
        return create_sample_data()

def create_sample_data():
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
    
    for _ in range(1000):
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
            
            if np.random.random() < 0.3:
                noise_options = [
                    f"{post} #staySafe",
                    f"{post} omg",
                    f"{post} this is crazy",
                    f"{post} pls be careful",
                    f"{post} 🙏",
                    f"{post} #emergency",
                    f"{post} wtf",
                    f"{post} this is insane"
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
            
            if np.random.random() < 0.4:
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
                    f"{post} this is insane"
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
    
    noisy_posts = []
    noisy_labels = []
    
    for i, post in enumerate(all_posts):
        if np.random.random() < 0.2:
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
        
        if np.random.random() < 0.1:
            post = post + " xyz"
        
        if np.random.random() < 0.15:
            post = post.replace(" ", "  ")
        
        noisy_posts.append(post)
        noisy_labels.append(labels[i])
    
    all_posts = noisy_posts
    labels = noisy_labels
    
    return pd.DataFrame({
        'text': all_posts,
        'label': labels
    })


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'memory_usage_mb': memory_usage
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    print(f"Training Time: {training_time:.4f}s")
    print(f"Prediction Time: {prediction_time:.4f}s")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    
    return results

def main():
    print("BlueSky Disaster Detection - ML Model Comparison")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    print("\nLoading disaster dataset...")
    df = load_real_disaster_data()
    print(f"Dataset size: {len(df)} posts")
    print(f"Disaster posts: {sum(df['label'])}")
    print(f"Non-disaster posts: {len(df) - sum(df['label'])}")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} posts")
    print(f"Test set: {len(X_test)} posts")
    
    print("\nVectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(random_state=42, kernel='linear'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1)
    }
    
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_train_vec, X_test_vec, y_train, y_test, name)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print(f"\n{'='*80}")
    print("FINAL MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    results_df.to_csv('ml_model_comparison_results.csv', index=False)
    print(f"\nResults saved to: ml_model_comparison_results.csv")
    
    formatted_df = results_df.copy()
    
    for col in ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) and x > 0 else "N/A")
    
    if 'cv_std' in formatted_df.columns:
        formatted_df['cv_std'] = formatted_df['cv_std'].apply(lambda x: f"± {x*100:.2f}%" if pd.notna(x) and x > 0 else "")
    
    if 'cv_mean' in formatted_df.columns and 'cv_std' in formatted_df.columns:
        formatted_df['CV Score'] = formatted_df['cv_mean'] + " " + formatted_df['cv_std']
        formatted_df = formatted_df.drop(['cv_mean', 'cv_std'], axis=1)
    
    for col in ['training_time', 'prediction_time']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
    
    if 'memory_usage_mb' in formatted_df.columns:
        formatted_df['memory_usage_mb'] = formatted_df['memory_usage_mb'].apply(lambda x: f"{x:.2f}")
    
    formatted_df.columns = formatted_df.columns.str.replace('_', ' ').str.title()
    formatted_df.columns = formatted_df.columns.str.replace('Mb', 'MB')
    formatted_df.columns = formatted_df.columns.str.replace('Time', 'Time (s)')
    
    formatted_df.to_csv('ml_model_comparison_results_formatted.csv', index=False)
    print(f"Formatted results saved to: ml_model_comparison_results_formatted.csv")
    
    best_model = results_df.iloc[0]
    print(f"\nBEST MODEL: {best_model['model']}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   F1-Score: {best_model['f1_score']:.4f}")
    print(f"   Training Time: {best_model['training_time']:.4f}s")
    
    print(f"\nComparison completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
