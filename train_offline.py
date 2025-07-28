#!/usr/bin/env python3
"""
Offline training script using sklearn for Thai fake news classification
Works without internet connection or Hugging Face authentication
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import os
import re

def clean_thai_text(text):
    """Clean Thai text for training"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove digits
    text = re.sub(r'[0-9]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("🚀 Starting Offline Thai Fake News Classification Training")
    print("=" * 60)
    
    # Load prepared data
    print("📊 Loading training data...")
    try:
        X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
        X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
        y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
        y_test = np.load('data/processed/y_test.npy', allow_pickle=True)
        label_encoder = joblib.load('data/processed/label_encoder.pkl')
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        print(f"✓ Label classes: {label_encoder.classes_}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run: python src/core/data_preparation.py")
        return
    
    # Clean the text data
    print("\n🧹 Cleaning text data...")
    X_train_clean = [clean_thai_text(text) for text in X_train]
    X_test_clean = [clean_thai_text(text) for text in X_test]
    
    # Define models to try
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', SVC(kernel='linear', random_state=42, probability=True))
        ])
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}
    
    print("\n🏋️ Training multiple models...")
    
    for name, model in models.items():
        print(f"\n📈 Training {name}...")
        
        # Train the model
        model.fit(X_train_clean, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_clean)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ {name} Accuracy: {accuracy:.4f}")
        
        # Store results
        results[name] = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=label_encoder.classes_, 
                                                         output_dict=True)
        }
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_accuracy:.4f})")
    
    # Generate detailed report for best model
    print(f"\n📋 Detailed Results for {best_name}:")
    y_pred_best = best_model.predict(X_test_clean)
    print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
    
    # Save the best model
    output_dir = "models/offline-thai-fakenews-classifier"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving {best_name} model to {output_dir}...")
    
    # Save model
    joblib.dump(best_model, f'{output_dir}/model.pkl')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    
    # Save model info
    model_info = {
        'model_type': 'sklearn_pipeline',
        'model_name': best_name,
        'accuracy': float(best_accuracy),
        'label_classes': label_encoder.classes_.tolist(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_extraction': 'TfidfVectorizer',
        'max_features': 5000,
        'ngram_range': [1, 2]
    }
    
    with open(f'{output_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save all results
    with open(f'{output_dir}/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Model saved successfully!")
    print(f"✅ Model info: {output_dir}/model_info.json")
    print(f"✅ All results: {output_dir}/all_results.json")
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    
    # Load saved model
    saved_model = joblib.load(f'{output_dir}/model.pkl')
    saved_label_encoder = joblib.load(f'{output_dir}/label_encoder.pkl')
    
    # Test samples
    test_samples = [
        "รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า",
        "พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง",
        "วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์",
        "พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%"
    ]
    
    print("\n🔮 Sample Predictions:")
    print("-" * 80)
    
    for i, text in enumerate(test_samples, 1):
        cleaned_text = clean_thai_text(text)
        prediction = saved_model.predict([cleaned_text])[0]
        probabilities = saved_model.predict_proba([cleaned_text])[0]
        predicted_label = saved_label_encoder.classes_[prediction]
        confidence = max(probabilities)
        
        print(f"{i}. Text: {text[:50]}...")
        print(f"   Prediction: {predicted_label} (Confidence: {confidence:.3f})")
        
        # Show probabilities for each class
        prob_str = ", ".join([f"{cls}={prob:.3f}" for cls, prob in 
                             zip(saved_label_encoder.classes_, probabilities)])
        print(f"   Probabilities: {prob_str}")
        print()
    
    print("🎊 Training completed successfully!")
    print(f"📁 Find your trained model in: {output_dir}")
    print("\n🚀 Next steps:")
    print("   - Update predictor to use this model")
    print("   - Run app: streamlit run src/app.py")

if __name__ == "__main__":
    main()