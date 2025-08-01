import torch
import numpy as np
import json
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsPredictor:
    """
    Thai Fake News Predictor using fine-tuned WangchanBERTa model
    """
    
    def __init__(self, model_path: str = "models/wangchanberta-finetuned-afnc"):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.model_info = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model components
        self._load_model()
        
    def _load_model(self):
        """Load the trained model, tokenizer, and metadata"""
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Load model info
            model_info_path = os.path.join(self.model_path, 'model_info.json')
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    self.model_info = json.load(f)
                logger.info(f"Loaded model info: {self.model_info['model_name']}")
            else:
                logger.warning("Model info not found, using defaults")
                self.model_info = {'max_length': 256, 'num_labels': 2}
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("✓ Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model loaded successfully on {self.device}")
            
            # Load label encoder
            label_encoder_path = os.path.join(self.model_path, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info(f"✓ Label encoder loaded: {self.label_encoder.classes_}")
            else:
                # Fallback labels
                logger.warning("Label encoder not found, using default labels")
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.array(['Fake', 'Real'])
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str, return_probabilities: bool = True) -> Dict[str, any]:
        """
        Predict if a news headline is real or fake
        
        Args:
            text (str): Thai news headline text
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            Dict containing prediction, confidence, and optionally probabilities
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Tokenize
            max_length = self.model_info.get('max_length', 256)
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get prediction
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                predicted_label = self.label_encoder.classes_[predicted_class_idx]
                
                # Prepare result
                result = {
                    'prediction': predicted_label,
                    'confidence': float(confidence),
                    'is_fake': predicted_label == 'Fake'
                }
                
                if return_probabilities:
                    prob_dict = {}
                    for i, class_name in enumerate(self.label_encoder.classes_):
                        prob_dict[class_name] = float(probabilities[0][i].item())
                    result['probabilities'] = prob_dict
                
                return result
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict multiple texts at once
        
        Args:
            texts (List[str]): List of Thai news headlines
            
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []
        
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text '{text[:50]}...': {e}")
                results.append({
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'is_fake': None,
                    'error': str(e)
                })
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """
        Clean input text (basic preprocessing)
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model metadata
        """
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'model_info': self.model_info,
            'label_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None,
            'is_loaded': self.model is not None
        }

# Convenience functions for direct usage
def load_predictor(model_path: str = "models/wangchanberta-finetuned-afnc") -> FakeNewsPredictor:
    """
    Load a fake news predictor
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        FakeNewsPredictor instance
    """
    return FakeNewsPredictor(model_path)

def predict_news(text: str, model_path: str = "models/wangchanberta-finetuned-afnc") -> Dict[str, any]:
    """
    Quick prediction function
    
    Args:
        text (str): Thai news headline
        model_path (str): Path to the trained model
        
    Returns:
        Dict containing prediction results
    """
    predictor = load_predictor(model_path)
    return predictor.predict(text)

# Test function
def test_predictor():
    """Test the predictor with sample data"""
    try:
        # Test samples (English for testing - in production use Thai)
        test_samples = [
            "Government announces economic development plan for next year",  # Real news
            "New effective diabetes medication discovered by researchers",  # Real news  
            "Scientists find miracle herb that helps lose weight in 1 week",  # Fake news
            "Coconut oil found to cure cancer with 100% effectiveness"  # Fake news
        ]
        
        print("Testing Fake News Predictor...")
        print("=" * 50)
        
        # Load predictor
        predictor = load_predictor()
        print(f"Model loaded successfully: {predictor.get_model_info()}")
        print()
        
        # Test predictions
        for i, text in enumerate(test_samples, 1):
            result = predictor.predict(text)
            print(f"{i}. Text: {text[:50]}...")
            print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
            if 'probabilities' in result:
                print(f"   Probabilities: {result['probabilities']}")
            print()
        
        # Test batch prediction
        print("Testing batch prediction...")
        batch_results = predictor.predict_batch(test_samples[:2])
        print(f"Batch prediction completed: {len(batch_results)} results")
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_predictor()