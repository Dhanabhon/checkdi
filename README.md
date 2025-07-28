# CheckDi: Thai Fake News Verification Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

CheckDi is an AI-powered fake news verification assistant specifically designed for the Thai language. Leveraging state-of-the-art natural language processing techniques and real data from Thailand's Anti-Fake News Center (AFNC), CheckDi helps users identify potentially misleading information in Thai news content.

This project demonstrates a comprehensive AI development pipeline, from data collection and preprocessing to model training and deployment of a user-friendly web application.

## Key Features

- **News Classification**: Utilizes a fine-tuned WangchanBERTa model to predict whether input text is "Real News" or "Fake News"
- **Confidence Scoring**: Provides probability scores to indicate the model's confidence in its predictions
- **Explainable AI (XAI)**: Highlights key words and phrases that influenced the model's decision, enhancing transparency
- **Intuitive Web Interface**: Built with Streamlit for a seamless user experience

## Architecture

CheckDi follows a 3-tier architecture for clear separation of concerns:

1. **Presentation Tier (Frontend)**: Streamlit-based interface for user interaction
2. **Logic Tier (Backend)**: AI model and prediction logic processing
3. **Data Tier**: Dataset from AFNC and trained model files

## Technology Stack

- **Language**: Python 3.9+
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Web Scraping**: BeautifulSoup4, Requests
- **AI/NLP**: PyTorch, Hugging Face Transformers, wangchanberta-base-att-spm-uncased
- **Web Framework**: Streamlit
- **Development Tools**: Jupyter Notebook, Visual Studio Code

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/checkdi.git
   cd checkdi
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it (choose based on your OS)
   .\venv\Scripts\activate    # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

The project includes a real dataset from the Anti-Fake News Center (AFNC) with over 2,700 news items. The data preparation process:

1. Loads and processes the AFNC dataset, categorizing news as "Real" or "Fake"
2. Cleans and preprocesses the text data
3. Splits the data into training and testing sets

Run the data preparation:
```bash
python src/core/data_preparation.py
```

### Data Visualization

The project includes visualization tools to understand the prepared data:

1. Label distribution charts
2. Text length analysis
3. Word clouds for real vs fake news
4. Data preparation pipeline visualization

Run the visualization script:
```bash
python src/core/visualize_data.py
```

Alternatively, use the Jupyter notebook for a more interactive approach:
```
notebooks/03_data_visualization.ipynb
```

### Model Training

Train the model using:
```
notebooks/04_model_training.ipynb
```
_Note: This step can be time-consuming and may require a GPU for optimal performance._

### Running the Application

Start the Streamlit application:
```bash
streamlit run src/app.py
```

Access the application in your browser at `http://localhost:8501`.

## Project Structure

```
checkdi/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   ├── raw/
│   │   └── AFNC_Opendata_export_20250728184932.csv
│   └── visualization/
├── models/
│   └── wangchanberta-finetuned-afnc/
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 03_data_visualization.ipynb
│   └── 04_model_training.ipynb
└── src/
    ├── app.py
    └── core/
        ├── __init__.py
        ├── data_preparation.py
        ├── visualize_data.py
        ├── predictor.py
        └── scraper.py
```

## Future Improvements

- **URL-based Verification**: Enable users to paste news URLs for automatic content fetching and verification
- **Model Enhancement**: Experiment with alternative Thai language models and data augmentation techniques
- **Database Expansion**: Incorporate data from additional sources to improve model accuracy across news categories
- **Cloud Deployment**: Deploy the application to cloud platforms (e.g., Hugging Face Spaces, Heroku, AWS) for public access

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Tom Dhanabhon