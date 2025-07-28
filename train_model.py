#!/usr/bin/env python3
"""
Script to train WangchanBERTa model for Thai fake news detection
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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
    print("🚀 Starting WangchanBERTa Training for Thai Fake News Detection")
    print("=" * 60)
    
    # Configuration
    # Try Thai models first, fallback to multilingual models
    MODEL_OPTIONS = [
        "pythainlp/thai-roberta-base",  # Thai RoBERTa from PyThaiNLP
        "facebook/xlm-roberta-base",  # Multilingual model that works with Thai
        "bert-base-multilingual-cased",  # Alternative multilingual option
        "airesearch/wangchanberta-base-att-spm-uncased"  # Original if available
    ]
    
    MODEL_NAME = None
    for model_option in MODEL_OPTIONS:
        try:
            print(f"🔍 Trying model: {model_option}")
            # Test if we can access the model
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_option)
            MODEL_NAME = model_option
            print(f"✅ Successfully connected to: {model_option}")
            break
        except Exception as e:
            print(f"❌ Cannot access {model_option}: {str(e)[:100]}...")
            continue
    
    if MODEL_NAME is None:
        print("❌ Cannot access any supported models. Please check your internet connection.")
        return
    OUTPUT_DIR = "models/wangchanberta-finetuned-afnc"
    MAX_LENGTH = 256
    BATCH_SIZE = 16  # Adjust based on GPU memory
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Training on CPU will be slower. Consider using GPU for better performance.")
    
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
        # Load splits
        X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
        X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
        y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
        y_test = np.load('data/processed/y_test.npy', allow_pickle=True)
        
        # Load label encoder
        label_encoder = joblib.load('data/processed/label_encoder.pkl')
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        print(f"✓ Label classes: {label_encoder.classes_}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run data preparation first: python src/core/data_preparation.py")
        return
    
    # Load tokenizer and model
    print("\n🤖 Loading WangchanBERTa model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification"
    )
    
    # Move model to device
    model.to(device)
    print(f"✓ Model loaded with {model.num_labels} output labels")
    print(f"✓ Model moved to {device}")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = NewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = NewsDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard
        push_to_hub=False
    )
    
    # Create trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\n🏋️  Starting model training...")
    print("Note: This may take several minutes to hours depending on your hardware.")
    
    # Train
    train_result = trainer.train()
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training runtime: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
    
    # Evaluate the model
    print("\n📈 Evaluating model on test set...")
    eval_result = trainer.evaluate()
    
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"Test loss: {eval_result['eval_loss']:.4f}")
    
    # Generate detailed predictions and metrics
    print("\n🔍 Generating detailed evaluation metrics...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Classification report
    target_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Save the final model and tokenizer
    print("\n💾 Saving final model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save additional artifacts
    model_info = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'num_labels': len(label_encoder.classes_),
        'label_classes': label_encoder.classes_.tolist(),
        'final_accuracy': eval_result['eval_accuracy'],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'training_loss': train_result.training_loss,
        'eval_loss': eval_result['eval_loss']
    }
    
    with open(f'{OUTPUT_DIR}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save detailed results
    results = {
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg'],
        'per_class': {target_names[i]: report[target_names[i]] for i in range(len(target_names))}
    }
    
    with open(f'{OUTPUT_DIR}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save the label encoder
    joblib.dump(label_encoder, f'{OUTPUT_DIR}/label_encoder.pkl')
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    print(f"✅ Model info saved to {OUTPUT_DIR}/model_info.json")
    print(f"✅ Evaluation results saved to {OUTPUT_DIR}/evaluation_results.json")
    print(f"✅ Label encoder saved to {OUTPUT_DIR}/label_encoder.pkl")
    
    # Test the saved model with sample predictions
    print("\n🧪 Testing saved model with sample predictions...")
    
    # Load the saved model
    saved_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    saved_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    saved_model.to(device)
    saved_model.eval()
    
    # Test samples (Thai news headlines)
    test_samples = [
        "รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า",  # Real news
        "พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง",  # Real news
        "วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์",  # Fake news
        "พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%"  # Fake news
    ]
    
    print("\n🔮 Sample Predictions:")
    print("-" * 80)
    
    for i, text in enumerate(test_samples, 1):
        # Tokenize
        inputs = saved_tokenizer(
            text, 
            return_tensors="pt", 
            max_length=MAX_LENGTH, 
            truncation=True, 
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = saved_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_label = label_encoder.classes_[predicted_class]
        
        print(f"{i}. Text: {text[:50]}...")
        print(f"   Prediction: {predicted_label} (Confidence: {confidence:.3f})")
        print(f"   Probabilities: Real={probabilities[0][1]:.3f}, Fake={probabilities[0][0]:.3f}")
        print()
    
    print("🎊 Model training completed successfully!")
    print(f"📁 Find your trained model in: {OUTPUT_DIR}")
    print("\n🚀 Next steps:")
    print("   - Test the model: python -c \"from src.core.predictor import test_predictor; test_predictor()\"")
    print("   - Run the app: streamlit run src/app.py")

if __name__ == "__main__":
    main()