import joblib
import json
import os
import re
import logging
from typing import Dict, List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineFakeNewsPredictor:
    """
    Offline Thai Fake News Predictor using sklearn models
    """
    
    def __init__(self, model_path: str = "models/offline-thai-fakenews-classifier"):
        """
        Initialize the predictor with a trained sklearn model
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.model_info = None
        
        # Load model components
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and metadata"""
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
                logger.warning("Model info not found")
                self.model_info = {}
            
            # Load model
            model_file = os.path.join(self.model_path, 'model.pkl')
            self.model = joblib.load(model_file)
            logger.info("✓ Model loaded successfully")
            
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
    
    def _clean_text(self, text: str) -> str:
        """Clean Thai text for prediction"""
        if not text:
            return ""
        
        text = str(text).strip()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove digits
        text = re.sub(r'[0-9]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
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
            
            # Predict
            prediction = self.model.predict([cleaned_text])[0]
            probabilities = self.model.predict_proba([cleaned_text])[0]
            
            # Get prediction details
            predicted_label = self.label_encoder.classes_[prediction]
            confidence = float(max(probabilities))
            
            # Prepare result
            result = {
                'prediction': predicted_label,
                'confidence': confidence,
                'is_fake': predicted_label == 'Fake'
            }
            
            if return_probabilities:
                prob_dict = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    prob_dict[class_name] = float(probabilities[i])
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
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model metadata
        """
        return {
            'model_path': self.model_path,
            'model_info': self.model_info,
            'label_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None,
            'is_loaded': self.model is not None,
            'model_type': 'sklearn_offline'
        }

# Convenience functions
def load_predictor(model_path: str = "models/offline-thai-fakenews-classifier") -> OfflineFakeNewsPredictor:
    """Load an offline fake news predictor"""
    return OfflineFakeNewsPredictor(model_path)

def predict_news(text: str, model_path: str = "models/offline-thai-fakenews-classifier") -> Dict[str, any]:
    """Quick prediction function"""
    predictor = load_predictor(model_path)
    return predictor.predict(text)

# Test function
def test_predictor():
    """Test the offline predictor"""
    try:
        # Test samples
        test_samples = [
            "รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า",
            "พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง",
            "วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์",
            "พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%"
        ]
        
        print("Testing Offline Fake News Predictor...")
        print("=" * 50)
        
        # Load predictor
        predictor = load_predictor()
        print(f"Model info: {predictor.get_model_info()}")
        print()
        
        # Test predictions
        for i, text in enumerate(test_samples, 1):
            result = predictor.predict(text)
            print(f"{i}. Text: {text[:50]}...")
            print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
            if 'probabilities' in result:
                print(f"   Probabilities: {result['probabilities']}")
            print()
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_predictor()