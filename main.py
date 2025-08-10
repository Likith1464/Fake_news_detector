# main.py - Main entry point for Fake News Detection System
"""
Run this file to start the fake news detection system
Press F5 in VS Code or run: python main.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.is_trained = False
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def create_sample_data(self):
        """Create sample training data"""
        print("🔧 Creating sample training data...")
        
        sample_articles = [
            # Real news examples
            "Scientists at Stanford University published research in the journal Nature showing promising results for a new cancer treatment approach. The peer-reviewed study followed 200 patients over 18 months.",
            "The Federal Reserve announced a 0.25% interest rate increase following today's meeting. Economic analysts had predicted this move based on recent inflation data.",
            "Local authorities report successful implementation of new traffic safety measures. The initiative, developed in partnership with city planners, aims to reduce pedestrian accidents.",
            "According to the World Health Organization, vaccination rates have increased by 15% globally this quarter. The organization credits improved distribution networks.",
            "Research published in the American Journal of Medicine indicates potential benefits of Mediterranean diet for heart health. The study analyzed data from 10,000 participants.",
            "The stock market closed 2.3% higher today following positive quarterly earnings reports from major technology companies.",
            "Climate scientists from NOAA released their annual report on global temperature trends. The peer-reviewed findings will inform upcoming policy discussions.",
            "University researchers develop new water purification technology. The innovation, funded by the National Science Foundation, could help communities access clean water.",
            
            # Fake news examples  
            "BREAKING: Doctors HATE this one weird trick that cures cancer in 24 hours! Big Pharma doesn't want you to know this SECRET method!",
            "SHOCKING: Government hiding ALIENS in Area 51! Former employee reveals TRUTH they don't want you to know!",
            "URGENT: This miracle fruit burns fat while you sleep! You won't BELIEVE the results! Click here NOW!",
            "EXPOSED: Vaccines contain MIND CONTROL chips! Secret documents leaked! Share before it's DELETED!",
            "AMAZING: Local mom discovers simple trick to make $5000 per day from home! Banks HATE her!",
            "WARNING: Your phone is SPYING on you! This one setting change will SHOCK you!",
            "INCREDIBLE: Scientists discover fountain of youth! This simple trick will make you look 20 years younger!",
            "MUST READ: Celebrity deaths predicted by time traveler! You won't believe what happens next!"
        ]
        
        labels = ['real'] * 8 + ['fake'] * 8
        
        df = pd.DataFrame({
            'text': sample_articles,
            'label': labels
        })
        
        # Save to CSV
        df.to_csv('data/sample_training_data.csv', index=False)
        print(f"✅ Sample data created: {len(df)} articles")
        print(f"📊 Distribution: {df['label'].value_counts().to_dict()}")
        return df
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs, emails, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_data(self, file_path=None, text_col='text', label_col='label'):
        """Load training data"""
        if file_path is None:
            # Use sample data
            if not os.path.exists('data/sample_training_data.csv'):
                self.create_sample_data()
            file_path = 'data/sample_training_data.csv'
        
        print(f"📚 Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            print("🔧 Creating sample data instead...")
            df = self.create_sample_data()
        
        # Validate columns
        if text_col not in df.columns or label_col not in df.columns:
            print(f"❌ Required columns '{text_col}', '{label_col}' not found")
            print(f"Available columns: {list(df.columns)}")
            return None, None
        
        # Preprocess text
        print("🔧 Preprocessing text...")
        df['processed_text'] = df[text_col].apply(self.preprocess_text)
        
        # Convert labels to binary
        df['binary_label'] = df[label_col].apply(
            lambda x: 1 if str(x).lower() in ['fake', 'false', '1'] else 0
        )
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        X = df['processed_text'].values
        y = df['binary_label'].values
        
        print(f"✅ Data loaded: {len(X)} samples")
        return X, y
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        print("\n🤖 Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        print("🔧 Creating text features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train multiple models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
        
        results = {}
        print("\n📈 Training Results:")
        print("-" * 50)
        
        for name, model in models_config.items():
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model
            self.models[name.lower().replace(' ', '_')] = model
            results[name] = accuracy
            
            print(f"{name:<20}: {accuracy:.4f} ({accuracy:.1%})")
        
        # Save models
        self.save_models()
        self.is_trained = True
        
        print(f"\n🏆 Best Model: {max(results, key=results.get)} ({max(results.values()):.1%})")
        return results
    
    def predict(self, text, model_name='logistic_regression'):
        """Predict if text is fake news"""
        if not self.is_trained:
            if not self.load_models():
                print("❌ No trained models found! Please train first.")
                return None
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'prediction': 'Unable to analyze',
                'confidence': 0.0,
                'is_fake': None
            }
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Get model
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        result = {
            'prediction': 'FAKE NEWS' if prediction == 1 else 'REAL NEWS',
            'confidence': confidence,
            'is_fake': prediction == 1,
            'model_used': model_name
        }
        
        return result
    
    def save_models(self):
        """Save trained models"""
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'is_trained': self.is_trained
        }
        
        with open('models/fake_news_detector.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("💾 Models saved to: models/fake_news_detector.pkl")
    
    def load_models(self):
        """Load trained models"""
        try:
            with open('models/fake_news_detector.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.models = model_data['models']
            self.is_trained = model_data['is_trained']
            
            print("✅ Models loaded successfully")
            return True
        except FileNotFoundError:
            print("❌ No saved models found")
            return False
    
    def analyze_text_interactive(self, text):
        """Interactive text analysis with detailed output"""
        print("\n" + "="*60)
        print("🔍 FAKE NEWS ANALYSIS RESULT")
        print("="*60)
        
        result = self.predict(text)
        
        if result is None:
            return
        
        print(f"📰 Text Preview: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🏷️  Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"🤖 Model Used: {result['model_used']}")
        
        if result['is_fake']:
            print("⚠️  WARNING: This text shows characteristics of fake news!")
            print("💡 Recommendation: Verify with reliable sources before sharing")
        else:
            print("✅ This text appears to be legitimate news content")
            print("💡 Note: Always cross-check important news with multiple sources")
        
        print("="*60)

def main_menu():
    """Main interactive menu"""
    detector = FakeNewsDetector()
    
    print("🚀 FAKE NEWS DETECTION SYSTEM")
    print("Welcome to VS Code Fake News Detector!")
    print("="*50)
    
    while True:
        print("\n📋 MAIN MENU:")
        print("1. 🏋️  Train New Model")
        print("2. 🔍 Analyze Text")
        print("3. 🧪 Test with Samples")
        print("4. 📊 Load Custom Dataset")
        print("5. 💾 Model Status")
        print("6. 🌐 Launch Web Interface")
        print("7. 🚪 Exit")
        
        choice = input("\n👉 Enter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\n🏋️  TRAINING NEW MODEL")
            print("-" * 30)
            X, y = detector.load_data()
            if X is not None:
                detector.train_models(X, y)
                print("✅ Training completed!")
        
        elif choice == '2':
            print("\n🔍 ANALYZE TEXT")
            print("-" * 20)
            text = input("📝 Enter news text to analyze:\n").strip()
            if text:
                detector.analyze_text_interactive(text)
            else:
                print("❌ Please enter some text!")
        
        elif choice == '3':
            print("\n🧪 TESTING WITH SAMPLE TEXTS")
            print("-" * 35)
            
            test_samples = [
                ("Scientists at MIT announce breakthrough in quantum computing research, published in Nature journal.", "Real News Example"),
                ("SHOCKING: This one weird trick will make you rich overnight! Banks hate this secret!", "Fake News Example"),
                ("The Federal Reserve maintains current interest rates following economic committee meeting.", "Real News Example")
            ]
            
            for text, description in test_samples:
                print(f"\n🧪 Testing: {description}")
                print(f"Text: {text}")
                result = detector.predict(text)
                if result:
                    print(f"Result: {result['prediction']} (Confidence: {result['confidence']:.1%})")
                print("-" * 40)
        
        elif choice == '4':
            print("\n📊 LOAD CUSTOM DATASET")
            print("-" * 25)
            file_path = input("📁 Enter CSV file path: ").strip()
            text_col = input("📝 Text column name (default: 'text'): ").strip() or 'text'
            label_col = input("🏷️  Label column name (default: 'label'): ").strip() or 'label'
            
            X, y = detector.load_data(file_path, text_col, label_col)
            if X is not None:
                train_now = input("🏋️  Train model now? (y/n): ").strip().lower()
                if train_now in ['y', 'yes']:
                    detector.train_models(X, y)
        
        elif choice == '5':
            print("\n💾 MODEL STATUS")
            print("-" * 20)
            if detector.load_models():
                print(f"✅ Models loaded: {len(detector.models)} algorithms")
                print(f"🤖 Available models: {list(detector.models.keys())}")
            else:
                print("❌ No trained models found")
                print("💡 Use option 1 to train a new model")
        
        elif choice == '6':
            print("\n🌐 LAUNCHING WEB INTERFACE")
            print("-" * 30)
            print("💡 Run this command in VS Code terminal:")
            print("   streamlit run web_app.py")
            print("📝 Make sure you've created web_app.py first!")
        
        elif choice == '7':
            print("\n👋 Thank you for using Fake News Detector!")
            print("🔒 Remember: Always verify news from multiple sources!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-7.")

if __name__ == "__main__":
    main_menu()