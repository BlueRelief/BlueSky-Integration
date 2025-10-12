import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import xgboost as xgb
import time
import psutil
import os
from datetime import datetime
import json

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
    
    return pd.DataFrame({
        'text': all_posts,
        'label': labels
    })

def get_model_param_grids():
    param_grids = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            }
        },
        'Naive Bayes': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                'fit_prior': [True, False]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced']
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
        }
    }
    
    return param_grids

def perform_grid_search(model, param_grid, X_train, y_train, model_name):
    print(f"\n{'='*70}")
    print(f"Grid Search for {model_name}")
    print(f"{'='*70}")
    print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    grid_search_time = time.time() - start_time
    
    print(f"\nGrid Search completed in {grid_search_time:.2f}s")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {json.dumps(grid_search.best_params_, indent=2)}")
    
    return grid_search, grid_search_time

def evaluate_best_model(grid_search, X_test, y_test, model_name, grid_search_time):
    print(f"\nEvaluating best {model_name} on test set...")
    
    best_model = grid_search.best_estimator_
    
    start_time = time.time()
    y_pred = best_model.predict(X_test)
    prediction_time = time.time() - start_time
    
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results = {
        'model': model_name,
        'best_params': grid_search.best_params_,
        'cv_best_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1_score': f1,
        'grid_search_time': grid_search_time,
        'prediction_time': prediction_time,
        'memory_usage_mb': memory_usage
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Prediction Time: {prediction_time:.4f}s")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    
    return results

def main():
    print("BlueSky Disaster Detection - Grid Search Hyperparameter Tuning")
    print("=" * 80)
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
    
    param_grids = get_model_param_grids()
    
    all_results = []
    best_params_dict = {}
    
    for model_name, config in param_grids.items():
        model = config['model']
        param_grid = config['params']
        
        grid_search, grid_search_time = perform_grid_search(
            model, param_grid, X_train_vec, y_train, model_name
        )
        
        results = evaluate_best_model(
            grid_search, X_test_vec, y_test, model_name, grid_search_time
        )
        
        all_results.append(results)
        best_params_dict[model_name] = grid_search.best_params_
    
    results_for_df = []
    for result in all_results:
        result_copy = result.copy()
        result_copy['best_params'] = json.dumps(result_copy['best_params'])
        results_for_df.append(result_copy)
    
    results_df = pd.DataFrame(results_for_df)
    results_df = results_df.sort_values('test_f1_score', ascending=False)
    
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS - BEST MODELS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    results_df.to_csv('ml_grid_search_results.csv', index=False)
    print(f"\nResults saved to: ml_grid_search_results.csv")
    
    with open('ml_best_hyperparameters.json', 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Best hyperparameters saved to: ml_best_hyperparameters.json")
    
    formatted_df = results_df.copy()
    
    for col in ['cv_best_score', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    
    for col in ['grid_search_time', 'prediction_time']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}s" if pd.notna(x) else "N/A")
    
    if 'memory_usage_mb' in formatted_df.columns:
        formatted_df['memory_usage_mb'] = formatted_df['memory_usage_mb'].apply(lambda x: f"{x:.2f} MB" if pd.notna(x) else "N/A")
    
    formatted_df.columns = formatted_df.columns.str.replace('_', ' ').str.title()
    
    formatted_df.to_csv('ml_grid_search_results_formatted.csv', index=False)
    print(f"Formatted results saved to: ml_grid_search_results_formatted.csv")
    
    print(f"\n{'='*80}")
    print("COMPARISON: Default vs Grid Search Results")
    print(f"{'='*80}")
    
    try:
        default_results = pd.read_csv('ml_model_comparison_results.csv')
        
        comparison_data = []
        for model_name in default_results['model'].values:
            default_row = default_results[default_results['model'] == model_name].iloc[0]
            
            grid_row_data = results_df[results_df['model'] == model_name]
            if not grid_row_data.empty:
                grid_row = grid_row_data.iloc[0]
                
                comparison_data.append({
                    'model': model_name,
                    'default_f1': default_row['f1_score'],
                    'grid_search_f1': grid_row['test_f1_score'],
                    'improvement': grid_row['test_f1_score'] - default_row['f1_score'],
                    'improvement_pct': ((grid_row['test_f1_score'] - default_row['f1_score']) / default_row['f1_score']) * 100
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('improvement', ascending=False)
        
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv('ml_grid_search_comparison.csv', index=False)
        print(f"\nComparison saved to: ml_grid_search_comparison.csv")
        
    except FileNotFoundError:
        print("Default results file not found. Skipping comparison.")
    
    best_model_result = all_results[0]
    print(f"\n{'='*80}")
    print("BEST MODEL OVERALL")
    print(f"{'='*80}")
    print(f"Model: {best_model_result['model']}")
    print(f"Best Parameters: {json.dumps(best_model_result['best_params'], indent=2)}")
    print(f"CV F1-Score: {best_model_result['cv_best_score']:.4f}")
    print(f"Test F1-Score: {best_model_result['test_f1_score']:.4f}")
    print(f"Test Accuracy: {best_model_result['test_accuracy']:.4f}")
    print(f"Grid Search Time: {best_model_result['grid_search_time']:.2f}s")
    
    print(f"\nGrid search completed at: {datetime.now()}")

if __name__ == "__main__":
    main()

