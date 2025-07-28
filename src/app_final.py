import streamlit as st
import sys
import os
from typing import Dict, Any
import time

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the offline predictor
try:
    from core.predictor_offline import OfflineFakeNewsPredictor, load_predictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="CheckDi - Thai Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Google-like styling
st.markdown("""
<style>
/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Google-like styling */
.main-title {
    text-align: center;
    font-size: 4rem;
    font-weight: 400;
    color: #202124;
    margin: 2rem 0 1rem 0;
}

.main-title .check {
    color: #4285f4;
}

.main-title .di {
    color: #ea4335;
}

.subtitle {
    text-align: center;
    color: #5f6368;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Search input styling */
.stTextInput > div > div > input {
    border: 1px solid #dfe1e5 !important;
    border-radius: 24px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    width: 100% !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus {
    border: 1px solid #4285f4 !important;
    box-shadow: 0 2px 8px rgba(66,133,244,0.3) !important;
}

/* Button styling */
.stButton > button {
    background-color: #f8f9fa !important;
    border: 1px solid #f8f9fa !important;
    border-radius: 4px !important;
    color: #3c4043 !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
    cursor: pointer !important;
    transition: box-shadow 0.2s !important;
}

.stButton > button:hover {
    box-shadow: 0 1px 1px rgba(0,0,0,.1) !important;
    background-color: #f8f9fa !important;
    border: 1px solid #dadce0 !important;
}

/* Result card styling */
.result-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid;
    color: #000000 !important;
}

.result-card h3 {
    color: #000000 !important;
}

.result-card p {
    color: #000000 !important;
}

.result-card strong {
    color: #000000 !important;
}

.result-card.fake {
    border-left-color: #ea4335;
    background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
}

.result-card.real {
    border-left-color: #34a853;
    background: linear-gradient(135deg, #f0fff4 0%, #ffffff 100%);
}

.confidence-bar {
    width: 100%;
    height: 8px;
    background-color: #e8eaed;
    border-radius: 4px;
    overflow: hidden;
    margin: 1rem 0;
}

.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.confidence-fill.fake {
    background: linear-gradient(90deg, #ea4335, #ff6b6b);
}

.confidence-fill.real {
    background: linear-gradient(90deg, #34a853, #4caf50);
}

/* Footer */
.footer {
    text-align: center;
    color: #5f6368;
    font-size: 0.9rem;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid #dadce0;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state and PREDICTOR_AVAILABLE:
    with st.spinner('Loading model...'):
        try:
            st.session_state.predictor = load_predictor()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.session_state.predictor = None

def predict_text(text: str) -> Dict[str, Any]:
    """Predict if text is fake or real news"""
    if not PREDICTOR_AVAILABLE or st.session_state.get('predictor') is None:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'is_fake': None,
            'error': 'Model not available'
        }
    
    try:
        return st.session_state.predictor.predict(text)
    except Exception as e:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'is_fake': None,
            'error': str(e)
        }

def display_result(result: Dict[str, Any], text: str):
    """Display prediction result in a card format"""
    if result.get('error'):
        st.error(f"Error: {result['error']}")
        return
    
    prediction = result['prediction']
    confidence = result['confidence']
    is_fake = result.get('is_fake', False)
    
    # Determine card style
    card_class = 'fake' if is_fake else 'real'
    emoji = '❌' if is_fake else '✅'
    label = 'Fake News' if is_fake else 'Real News'
    
    # Create result card
    st.markdown(f"""
    <div class="result-card {card_class}">
        <h3>{emoji} <span style="color: #000000;">{label}</span></h3>
        <p style="color: #000000;"><strong>Text:</strong> {text}</p>
        <p style="color: #000000;"><strong>Confidence:</strong> {confidence:.1%}</p>
        <div class="confidence-bar">
            <div class="confidence-fill {card_class}" style="width: {confidence * 100}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show detailed probabilities if available
    if 'probabilities' in result:
        probs = result['probabilities']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real News Probability", f"{probs.get('Real', 0):.1%}")
        with col2:
            st.metric("Fake News Probability", f"{probs.get('Fake', 0):.1%}")

def main():
    """Main application function"""
    
    # Check if model is available
    if not PREDICTOR_AVAILABLE:
        st.error("⚠️ Model not found. Please train the model first by running: `python train_offline.py`")
        st.stop()
    
    # Main title with Google-like styling
    st.markdown("""
    <div class="main-title">
        <span class="check">Check</span><span class="di">Di</span>
    </div>
    <div class="subtitle">Thai Fake News Detection Tool</div>
    """, unsafe_allow_html=True)
    
    # Search input (Google-like)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        search_text = st.text_input(
            "search",
            placeholder="Enter Thai news headline to verify...",
            label_visibility='collapsed'
        )
    
    # Buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_clicked = st.button("🔍 Analyze News", type='primary')
    
    # Example headlines
    st.markdown("### 📰 Example Headlines to Test:")
    
    examples = [
        "รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า",
        "พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง", 
        "วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์",
        "พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%",
        "นักวิทยาศาสตร์ค้นพบวิธีการใหม่ในการรักษาโรคหัวใจ"
    ]
    
    # Display examples with unique keys
    selected_example = None
    for i, example in enumerate(examples):
        if st.button(f"📄 {example[:50]}...", key=f"btn_example_{i}"):
            selected_example = example
    
    # Handle example selection
    if selected_example:
        search_text = selected_example
        analyze_clicked = True
    
    # Analysis logic
    if analyze_clicked and search_text and search_text.strip():
        st.markdown("### 📊 Analysis Results")
        
        # Show loading animation
        with st.spinner('Analyzing...'):
            time.sleep(1)  # Simulate processing time
            result = predict_text(search_text.strip())
        
        # Display results
        display_result(result, search_text)
        
        # Model information
        if not result.get('error'):
            with st.expander("🔧 Technical Details"):
                if PREDICTOR_AVAILABLE and st.session_state.get('predictor'):
                    model_info = st.session_state.predictor.get_model_info()
                    st.json(model_info)
    
    elif analyze_clicked and (not search_text or not search_text.strip()):
        st.warning("⚠️ Please enter text to analyze")
    
    # About section
    with st.expander("ℹ️ About CheckDi"):
        st.markdown("""
        **CheckDi** is a Thai fake news detection system powered by Machine Learning.
        
        **✨ Features:**
        - Fast analysis of Thai news headlines
        - 81.36% accuracy with SVM model
        - Confidence scores and probability breakdown
        - Easy-to-use Google-like interface
        
        **📋 How to use:**
        1. Enter a Thai news headline in the search box
        2. Click "🔍 Analyze News" button
        3. View the results with confidence scores
        4. Try example headlines by clicking the 📄 buttons
        
        **⚠️ Note:** This is a demonstration tool. Always verify information from reliable sources.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        CheckDi - Thai Fake News Detection System | Powered by Machine Learning ⚡
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()