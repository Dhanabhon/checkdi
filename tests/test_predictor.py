import unittest
import sys
import os
import tempfile
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.predictor import FakeNewsPredictor, load_predictor, predict_news

class TestFakeNewsPredictor(unittest.TestCase):
    """Test cases for the FakeNewsPredictor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_texts = [
            "Government announces economic development plan for next year",
            "New effective diabetes medication discovered",
            "Scientists find miracle weight loss herb works in 1 week", 
            "Coconut oil cures cancer 100 percent effectiveness found"
        ]
        
        # Create a mock model directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Create mock model_info.json
        model_info = {
            "model_name": "test-model",
            "max_length": 256,
            "num_labels": 2,
            "label_classes": ["Fake", "Real"],
            "final_accuracy": 0.95
        }
        with open(os.path.join(self.model_path, "model_info.json"), "w") as f:
            json.dump(model_info, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        predictor = FakeNewsPredictor.__new__(FakeNewsPredictor)
        
        # Test basic cleaning
        self.assertEqual(predictor._clean_text("  hello world  "), "hello world")
        self.assertEqual(predictor._clean_text("hello\n\nworld"), "hello world")
        self.assertEqual(predictor._clean_text(""), "")
        self.assertEqual(predictor._clean_text(None), "")
    
    @patch('src.core.predictor.AutoTokenizer')
    @patch('src.core.predictor.AutoModelForSequenceClassification')
    @patch('src.core.predictor.joblib.load')
    def test_model_loading(self, mock_joblib, mock_model, mock_tokenizer):
        """Test model loading process"""
        # Mock label encoder
        mock_label_encoder = MagicMock()
        mock_label_encoder.classes_ = np.array(['Fake', 'Real'])
        mock_joblib.return_value = mock_label_encoder
        
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Test initialization (should not raise exception)
        try:
            predictor = FakeNewsPredictor(self.model_path)
            self.assertIsNotNone(predictor.model_info)
            self.assertEqual(predictor.model_info['num_labels'], 2)
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs"""
        predictor = FakeNewsPredictor.__new__(FakeNewsPredictor)
        
        # Test empty string
        with self.assertRaises(ValueError):
            predictor.predict("")
        
        # Test None input
        with self.assertRaises(ValueError):
            predictor.predict(None)
        
        # Test whitespace only
        with self.assertRaises(ValueError):
            predictor.predict("   ")

class TestIntegration(unittest.TestCase):
    """Integration tests (require actual model)"""
    
    def test_model_exists(self):
        """Test if model directory exists"""
        model_path = "models/wangchanberta-finetuned-afnc"
        if os.path.exists(model_path):
            self.assertTrue(os.path.isdir(model_path))
        else:
            self.skipTest("Model not found - run training notebook first")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)