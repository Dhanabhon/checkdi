import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(file_path='data/processed/news_prepared.csv'):
    """Load the prepared data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from {file_path}")
    return df

def plot_label_distribution(df):
    """Plot the distribution of real vs fake news"""
    plt.figure(figsize=(10, 6))
    
    # Count plot
    label_counts = df['label'].value_counts()
    bars = plt.bar(label_counts.index, label_counts.values, color=['#ff6b6b', '#4ecdc4'])
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} ({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=12)
    
    plt.title('Distribution of News Labels', fontsize=16, pad=20)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/visualization/label_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_text_length_distribution(df):
    """Plot the distribution of text lengths"""
    df['headline_length'] = df['cleaned_headline'].astype(str).apply(len)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(df['headline_length'], bins=50, alpha=0.7, color='#6c5ce7')
    plt.title('Distribution of Headline Lengths', fontsize=14)
    plt.xlabel('Headline Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Box plot by label
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='label', y='headline_length', 
                hue='label', palette=['#ff6b6b', '#4ecdc4'], legend=False)
    plt.title('Headline Length by Label', fontsize=14)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Headline Length (characters)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('data/visualization/text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_wordclouds(df):
    """Generate word clouds for real and fake news"""
    # Create directory if it doesn't exist
    os.makedirs('data/visualization', exist_ok=True)
    
    # Try to find a suitable font for Thai text
    thai_fonts = [
        '/System/Library/Fonts/Thonburi.ttc',  # macOS
        '/usr/share/fonts/truetype/tlwg/Garuda.ttf',  # Linux
        'C:/Windows/Fonts/Tahoma.ttf',  # Windows
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # Alternative
    ]
    
    font_path = None
    for font in thai_fonts:
        if os.path.exists(font):
            font_path = font
            break
    
    if not font_path:
        print("Warning: Could not find a suitable font for Thai text. Word clouds may not display correctly.")
    
    for label in df['label'].unique():
        if pd.notna(label):
            # Get text for this label
            text = ' '.join(df[df['label'] == label]['cleaned_headline'].dropna())
            
            if text:
                # Generate word cloud with Thai font support
                wordcloud_params = {
                    'width': 800,
                    'height': 400,
                    'background_color': 'white',
                    'colormap': 'viridis',
                    'max_words': 100
                }
                
                # Add font path if found
                if font_path:
                    wordcloud_params['font_path'] = font_path
                
                wordcloud = WordCloud(**wordcloud_params).generate(text)
                
                # Plot
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {label} News', fontsize=16)
                
                # Save and show
                filename = f'data/visualization/wordcloud_{label.lower()}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print(f"No text data available for {label} news")

def plot_data_preparation_summary():
    """Plot a summary of the data preparation process"""
    # This would typically be generated after running the preparation script
    # For now, we'll use some example data
    
    # Sample data for demonstration
    stages = ['Raw Data', 'Text Cleaning', 'Train-Test Split', 'Final Dataset']
    counts = [291, 291, 291, 291]  # All the same in our case
    train_counts = [0, 0, 232, 232]
    test_counts = [0, 0, 59, 59]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars
    bar_width = 0.35
    index = np.arange(len(stages))
    
    bars1 = ax.bar(index - bar_width/2, train_counts, bar_width, 
                   label='Training Set', color='#4ecdc4', alpha=0.8)
    bars2 = ax.bar(index + bar_width/2, test_counts, bar_width, 
                   label='Test Set', color='#ff6b6b', alpha=0.8)
    
    # Add labels
    ax.set_xlabel('Data Preparation Stages', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title('Data Preparation Pipeline', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/visualization/data_preparation_pipeline.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all visualizations"""
    # Create visualization directory
    os.makedirs('data/visualization', exist_ok=True)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Generate visualizations
    print("Generating label distribution plot...")
    plot_label_distribution(df)
    
    print("Generating text length distribution plots...")
    plot_text_length_distribution(df)
    
    print("Generating word clouds...")
    generate_wordclouds(df)
    
    print("Generating data preparation pipeline visualization...")
    plot_data_preparation_summary()
    
    print("All visualizations saved to data/visualization/")

if __name__ == '__main__':
    main()