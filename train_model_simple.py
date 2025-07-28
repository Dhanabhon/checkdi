#!/usr/bin/env python3
"""
Simple training script using smaller, more practical models for Thai text classification
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import joblib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    print("🚀 Starting Thai Fake News Detection Model Training")
    print("=" * 60)
    
    # Configuration - using more practical models
    MODEL_OPTIONS = [
        "xlm-roberta-base",  # Multilingual RoBERTa (publicly available)
        "distilbert-base-multilingual-cased",  # Smaller, faster multilingual model
        "bert-base-multilingual-cased"  # Classic multilingual BERT
    ]
    
    OUTPUT_DIR = "models/thai-fakenews-classifier"
    MAX_LENGTH = 256
    BATCH_SIZE = 8  # Smaller batch size for stability
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5  # More epochs for better training
    WARMUP_STEPS = 100
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to find an available model
    MODEL_NAME = None
    for model_option in MODEL_OPTIONS:
        try:
            print(f"🔍 Trying model: {model_option}")
            # Test if we can access the model
            tokenizer_test = AutoTokenizer.from_pretrained(model_option)
            MODEL_NAME = model_option
            print(f"✅ Successfully connected to: {model_option}")
            break
        except Exception as e:
            print(f"❌ Cannot access {model_option}: {str(e)[:100]}...")
            continue
    
    if MODEL_NAME is None:
        print("❌ Cannot access any models. Using offline fallback...")
        print("💡 You can train without internet by downloading models manually")
        return
    
    print(f"\n📋 Training Configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load prepared data
    print("\n📊 Loading prepared data...")
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
    
    # Load tokenizer and model
    print(f"\n🤖 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification"
    )
    
    model.to(device)
    print(f"✓ Model loaded successfully")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = NewsDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=None,
        push_to_hub=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("\n🏋️ Starting training...")
    train_result = trainer.train()
    
    print("\n🎉 Training completed!")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    eval_result = trainer.evaluate()
    print(f"Final accuracy: {eval_result['eval_accuracy']:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save model info
    model_info = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'num_labels': len(label_encoder.classes_),
        'label_classes': label_encoder.classes_.tolist(),
        'final_accuracy': eval_result['eval_accuracy'],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(f'{OUTPUT_DIR}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    joblib.dump(label_encoder, f'{OUTPUT_DIR}/label_encoder.pkl')
    
    print(f"✅ Model saved to: {OUTPUT_DIR}")
    print("\n🚀 Next steps:")
    print("   - Test: python -c \"from src.core.predictor import test_predictor; test_predictor()\"")
    print("   - Run app: streamlit run src/app.py")

if __name__ == "__main__":
    main()