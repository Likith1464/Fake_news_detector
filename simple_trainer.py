# simple_trainer.py - Simplified training script for beginners
"""
This is a simplified version for quick training and testing
Run with: python simple_trainer.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import re
import os

def create_sample_data():
    """Create sample data for quick testing"""
    print("Creating sample training data...")
    
    # Sample real and fake news
    data = {
        'text': [
            # Real news
            "Scientists publish research in peer-reviewed journal about climate change effects",
            "Government announces new infrastructure spending plan after congressional approval",
            "Local hospital reports successful implementation of new patient safety protocols",
            "University researchers develop new water purification technology with NSF funding",
            "Federal Reserve adjusts interest rates following economic indicators analysis",
            
            # Fake news
            "BREAKING: This one weird trick doctors hate will cure everything instantly!",
            "SHOCKING secret that government doesn't want you to know about aliens!",
            "Miracle cure discovered but big pharma is hiding it from the public!",
            "Local mom makes $5000 per day with this simple trick banks hate!",
            "URGENT: Share this before it gets banned by the mainstream media!"
        ],
        'label': ['real', 'real', 'real', 'real', 'real', 'fake', 'fake', 'fake', 'fake', 'fake']
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)
    print(f"✅ Created sample_data.csv with {len(df)} articles")
    return df

def clean_text(text):
    """Simple text cleaning"""
    if pd.isna(text):
        return ''
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def train_simple_model():
    """Train a simple fake news detection model"""
    print("🤖 Training Fake News Detection Model")
    print("=" * 40)
    
    # Create sample data if it doesn't exist
    if not os.path.exists('sample_data.csv'):
        create_sample_data()
    
    # Load data
    df = pd.read_csv('sample_data.csv')
    print(f"📚 Loaded {len(df)} articles")
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Convert labels
    df['is_fake'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
    
    print(f"📊 Real news: {sum(df['is_fake'] == 0)}")
    print(f"📊 Fake news: {sum(df['is_fake'] == 1)}")
    
    # Prepare features
    X = df['cleaned_text'].values
    y = df['is_fake'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create features using TF-IDF
    print("🔧 Creating text features...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Train model
    print("🏋️ Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features, y_train)
    
    # Test accuracy
    predictions = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ Training completed!")
    print(f"🎯 Accuracy: {accuracy:.2%}")
    
    # Save model
    model_package = {
        'vectorizer': vectorizer,
        'model': model
    }
    
    with open('simple_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("💾 Model saved as 'simple_model.pkl'")
    return model_package

def predict_news(text, model_file='simple_model.pkl'):
    """Predict if news is fake or real"""
    
    # Load model
    try:
        with open(model_file, 'rb') as f:
            model_package = pickle.load(f)
        vectorizer = model_package['vectorizer']
        model = model_package['model']
    except FileNotFoundError:
        print("❌ Model not found! Training new model...")
        model_package = train_simple_model()
        vectorizer = model_package['vectorizer']
        model = model_package['model']
    
    # Clean and predict
    cleaned_text = clean_text(text)
    text_features = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_features)[0]
    probability = model.predict_proba(text_features)[0].max()
    
    # Show result
    result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    
    print("\n" + "="*50)
    print("🔍 ANALYSIS RESULT")
    print("="*50)
    print(f"📰 Text: {text[:80]}...")
    print(f"🏷️  Result: {result}")
    print(f"📊 Confidence: {probability:.1%}")
    
    if prediction == 1:
        print("⚠️  This appears to be fake news!")
    else:
        print("✅ This appears to be real news.")
    
    print("="*50)
    
    return {
        'is_fake': prediction == 1,
        'result': result,
        'confidence': probability
    }

def main():
    """Main function for interactive use"""
    print("🚀 SIMPLE FAKE NEWS DETECTOR")
    print("="*35)
    
    while True:
        print("\n📋 Options:")
        print("1. Train new model")
        print("2. Test news text")
        print("3. Quick test with samples")
        print("4. Exit")
        
        choice = input("\n👉 Choose option (1-4): ").strip()
        
        if choice == '1':
            train_simple_model()
        
        elif choice == '2':
            text = input("\n📝 Enter news text:\n")
            if text.strip():
                predict_news(text)
            else:
                print("❌ Please enter some text!")
        
        elif choice == '3':
            # Quick test samples
            samples = [
                "Scientists at Harvard published breakthrough research in Nature journal",
                "SHOCKING trick that doctors hate will cure everything in 24 hours!",
                "Government announces new policy after thorough committee review"
            ]
            
            print("\n🧪 Testing sample texts:")
            for i, sample in enumerate(samples, 1):
                print(f"\n--- Sample {i} ---")
                predict_news(sample)
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()