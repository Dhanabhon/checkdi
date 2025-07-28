# How to Train WangchanBERTa Model for Thai Fake News Detection

## 🚀 Quick Start

### Step 1: Prepare Environment
```bash
# Activate your conda environment
conda activate checkdi

# Ensure all dependencies are installed
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn wordcloud tqdm joblib
```

### Step 2: Prepare Data
```bash
# Run data preparation to create training splits
python src/core/data_preparation.py
```

### Step 3: Authenticate with Hugging Face
Choose one of these methods:

**Option A: Command Line Login**
```bash
# Install huggingface-hub
pip install huggingface-hub

# Login (you'll need a free Hugging Face account)
huggingface-cli login
```

**Option B: Set Environment Variable**
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

**Option C: Manual Token**
- Go to https://huggingface.co/settings/tokens
- Create a new token
- Set it in your environment

### Step 4: Train the Model

**Method 1: Command Line Training**
```bash
python train_model.py
```

**Method 2: Jupyter Notebook Training**
```bash
jupyter notebook notebooks/04_model_training.ipynb
```

## 📊 Training Configuration

The training uses these default settings:
- **Model**: WangchanBERTa (with fallbacks to multilingual models)
- **Max Length**: 256 tokens
- **Batch Size**: 16 (adjust for your GPU memory)
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Warmup Steps**: 100

## 🔧 Customization

To modify training parameters, edit these variables in `train_model.py`:

```python
MAX_LENGTH = 256        # Maximum sequence length
BATCH_SIZE = 16         # Batch size (reduce if GPU memory issues)
LEARNING_RATE = 2e-5    # Learning rate
NUM_EPOCHS = 3          # Number of training epochs
WARMUP_STEPS = 100      # Warmup steps for learning rate scheduler
```

## 📈 Expected Output

During training, you'll see:
1. Model loading and configuration
2. Training progress with loss values
3. Evaluation metrics (accuracy, loss)
4. Classification report with per-class metrics
5. Sample predictions on test data

## 📁 Output Files

After training, you'll find:
- `models/wangchanberta-finetuned-afnc/` - Complete trained model
- `models/wangchanberta-finetuned-afnc/model_info.json` - Model metadata
- `models/wangchanberta-finetuned-afnc/evaluation_results.json` - Detailed metrics
- `models/wangchanberta-finetuned-afnc/label_encoder.pkl` - Label encoder

## 🧪 Testing the Trained Model

```bash
# Test the predictor
python -c "from src.core.predictor import test_predictor; test_predictor()"

# Run the Streamlit app
streamlit run src/app.py
```

## 🚨 Troubleshooting

**Issue**: 401 Unauthorized Error
- **Solution**: Login to Hugging Face using `huggingface-cli login`

**Issue**: GPU Memory Error
- **Solution**: Reduce `BATCH_SIZE` from 16 to 8 or 4

**Issue**: Import Errors
- **Solution**: Install missing packages with `pip install package_name`

**Issue**: Data Not Found
- **Solution**: Run `python src/core/data_preparation.py` first

## ⚡ Performance Tips

1. **Use GPU**: Training on GPU is 10-50x faster than CPU
2. **Increase Batch Size**: Higher batch sizes train faster (if GPU memory allows)
3. **Monitor Memory**: Use `nvidia-smi` to monitor GPU memory usage
4. **Save Checkpoints**: Training automatically saves checkpoints every 100 steps

## 📚 Next Steps

After training:
1. Test the model with the predictor
2. Run the Streamlit web application
3. Experiment with different hyperparameters
4. Try data augmentation for better performance
5. Deploy the model to production

## 💡 Tips for Better Performance

1. **More Data**: Collect more Thai news data for better accuracy
2. **Data Quality**: Clean and verify your training data
3. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
4. **Cross-Validation**: Use k-fold cross-validation for robust evaluation