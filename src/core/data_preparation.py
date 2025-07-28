import pandas as pd
import numpy as np
import re
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Thai digits
    text = re.sub(r'[0-9]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespaces
    text = text.strip()
    
    return text

def prepare_afnc_data(raw_data_path='data/raw/AFNC_Opendata_export_20250728184932.csv',
                      processed_data_path='data/processed/news_prepared.csv'):
    """
    Load and prepare the AFNC dataset for model training
    
    Args:
        raw_data_path (str): Path to raw AFNC data CSV file
        processed_data_path (str): Path to save processed data CSV file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) split datasets
    """
    try:
        # Load raw data
        print("Loading AFNC data...")
        df = pd.read_csv(raw_data_path)
        print(f"✓ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Map categories to labels
        # Based on the data, we'll consider these categories as "Fake" news:
        # "ข่าวปลอม", "ข่าวบิดเบือน", "ข่าวลือ"
        # And these as "Real" news:
        # "ข่าวจริง", "คลังความรู้", "ข่าวอื่นๆ", "นโยบายรัฐบาล-ข่าวสาร", 
        # "ผลิตภัณฑ์สุขภาพ", "การเงิน-หุ้น", "อาชญากรรมออนไลน์", "ความสงบและความมั่นคง", "ภัยพิบัติ"
        # "กิจกรรม", "Feature News" (if present)
        
        fake_categories = ["ข่าวปลอม", "ข่าวบิดเบือน", "ข่าวลือ"]
        
        # Create label column based on category
        df['label'] = df['ประเภทข่าว'].apply(
            lambda x: 'Fake' if x in fake_categories else 'Real'
        )
        
        print("Data distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Use headline as text data
        df['headline'] = df['หัวข้อข่าว']
        
        # Clean headline text with progress bar
        print("Cleaning text data...")
        tqdm.pandas(desc="Cleaning headlines")
        df['cleaned_headline'] = df['headline'].progress_apply(clean_text)
        
        # Remove empty headlines
        initial_count = len(df)
        df = df[df['cleaned_headline'].notna() & (df['cleaned_headline'] != '')]
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} entries with empty headlines")
        
        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])
        
        # Select only relevant columns for saving
        df_processed = df[['headline', 'cleaned_headline', 'label', 'label_encoded']].copy()
        
        # Save prepared data
        df_processed.to_csv(processed_data_path, index=False)
        print(f"✓ Prepared data saved to {processed_data_path}")
        
        # Prepare features and labels for training
        X = df['cleaned_headline'].values
        y = df['label_encoded'].values
        
        # Split data
        print("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, label_encoder
        
    except FileNotFoundError:
        logger.error(f"File {raw_data_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise

def load_and_prepare_data(raw_data_path='data/raw/news.csv', 
                         processed_data_path='data/processed/news_prepared.csv'):
    """
    Load raw data and prepare it for model training
    
    Args:
        raw_data_path (str): Path to raw data CSV file
        processed_data_path (str): Path to save processed data CSV file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) split datasets
    """
    try:
        # Check if we have the AFNC dataset
        afnc_file = 'data/raw/AFNC_Opendata_export_20250728184932.csv'
        try:
            pd.read_csv(afnc_file, nrows=1)
            print("Using AFNC dataset for training")
            return prepare_afnc_data(afnc_file, processed_data_path)
        except FileNotFoundError:
            pass
        
        # Load raw data
        print(f"Loading data from {raw_data_path}...")
        df = pd.read_csv(raw_data_path)
        print(f"✓ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        print("Data distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Clean headline text with progress bar
        print("Cleaning text data...")
        tqdm.pandas(desc="Cleaning headlines")
        df['cleaned_headline'] = df['headline'].progress_apply(clean_text)
        
        # Remove empty headlines
        initial_count = len(df)
        df = df[df['cleaned_headline'].notna() & (df['cleaned_headline'] != '')]
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} entries with empty headlines")
        
        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])
        
        # Save prepared data
        df.to_csv(processed_data_path, index=False)
        print(f"✓ Prepared data saved to {processed_data_path}")
        
        # Prepare features and labels for training
        X = df['cleaned_headline'].values
        y = df['label_encoded'].values
        
        # Split data
        print("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, label_encoder
        
    except FileNotFoundError:
        logger.error(f"File {raw_data_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise

def create_sample_dataset():
    """
    Create a sample dataset for testing purposes
    """
    sample_data = {
        'headline': [
            "รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า",
            "พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง",
            "ผู้เชี่ยวชาญด้านสุขภาพเผยถึงอันตรายของการดื่มเครื่องดื่มแอลกอฮอล์",
            "การศึกษาใหม่แสดงให้เห็นว่าออกกำลังกายช่วยเพิ่มภูมิคุ้มกัน",
            "วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์",
            "หมอเผยผลข้างเคียงของการฉีดวัคซีน mRNA ที่ไม่เคยมีคนพูดถึงมาก่อน",
            "พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%",
            "การใช้สมาร์ทโฟนมากกว่า 2 ชั่วโมงต่อวันทำให้สมองย่อขนาดลง"
        ],
        'label': [
            'Real', 'Real', 'Real', 'Real',
            'Fake', 'Fake', 'Fake', 'Fake'
        ],
        'date': ['N/A'] * 8,
        'link': ['N/A'] * 8
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/raw/news.csv', index=False)
    print("✓ Sample dataset created at data/raw/news.csv")

def print_summary(X_train, X_test, y_train, y_test, label_encoder):
    """
    Print a detailed summary of the prepared data
    """
    print("\n" + "="*50)
    print("DATA PREPARATION SUMMARY")
    print("="*50)
    print(f"Total samples: {len(X_train) + len(X_test)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/(len(X_train) + len(X_test))*100:.1f}%)")
    print(f"Testing samples: {len(X_test)} ({len(X_test)/(len(X_train) + len(X_test))*100:.1f}%)")
    print(f"Label classes: {list(label_encoder.classes_)}")
    
    # Label distribution in training set
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining set label distribution:")
    for label, count in zip(unique, counts):
        class_name = label_encoder.classes_[label]
        print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Label distribution in test set
    unique, counts = np.unique(y_test, return_counts=True)
    print("\nTest set label distribution:")
    for label, count in zip(unique, counts):
        class_name = label_encoder.classes_[label]
        print(f"  {class_name}: {count} ({count/len(y_test)*100:.1f}%)")
    
    print("\nData preparation completed successfully!")
    print("="*50)

if __name__ == '__main__':
    # Create sample dataset if raw data doesn't exist
    try:
        pd.read_csv('data/raw/news.csv')
    except FileNotFoundError:
        print("Raw data not found, creating sample dataset...")
        create_sample_dataset()
    
    # Prepare data
    X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data()
    
    # Print detailed summary
    print_summary(X_train, X_test, y_train, y_test, label_encoder)