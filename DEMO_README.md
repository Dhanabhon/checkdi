# 🔍 CheckDi Demo - Thai Fake News Detection

A modern, Google-like web interface for detecting Thai fake news using Machine Learning.

## 🎯 Demo Features

### ✨ **Google-Inspired UI**
- **Clean, minimal design** similar to Google Search
- **Centered layout** with large, colorful CheckDi logo
- **Rounded search box** with focus animations
- **Card-based results** with confidence bars
- **Responsive design** that works on mobile and desktop

### 🌐 **Bilingual Support**
- **Thai Language**: Complete Thai interface with proper Thai text
- **English Language**: Full English interface for international users
- **Dynamic switching** between languages

### 🤖 **Smart Analysis**
- **Real-time prediction** with confidence scores
- **Visual confidence bars** showing prediction strength
- **Detailed probability breakdown** for both Real/Fake classifications
- **Technical model information** in expandable sections

### 📱 **User Experience**
- **Example headlines** to try with one click
- **Random example generator** for easy testing
- **Loading animations** for better user feedback
- **Error handling** with helpful messages

## 🚀 **How to Run the Demo**

### **Quick Start (Recommended)**
```bash
# 1. Navigate to project directory
cd /Users/dhanabhon/Projects/Git/checkdi

# 2. Run the launcher script
python run_app.py
```

### **Manual Launch**
```bash
# 1. Activate environment
conda activate checkdi

# 2. Ensure model is trained
python train_offline.py

# 3. Launch Streamlit app
streamlit run src/app.py
```

### **Access the Demo**
- **Local URL**: http://localhost:8501
- **Network URL**: Available to devices on your network

## 🎨 **UI Screenshots & Features**

### **Main Interface**
- **Large "CheckDi" logo** with Google-style colors (blue + red)
- **Subtitle** explaining the purpose in both languages
- **Search input** with rounded corners and hover effects
- **Action buttons** for analysis and examples

### **Language Toggle**
- **Top-right dropdown** to switch between 🇹🇭 Thai and 🇺🇸 English
- **Instant UI translation** of all text elements
- **Language-specific examples** that change automatically

### **Results Display**
- **Color-coded cards**: 
  - 🟢 **Green gradient** for Real News
  - 🔴 **Red gradient** for Fake News
- **Confidence bars** with smooth animations
- **Detailed metrics** showing probability percentages
- **Technical details** in expandable sections

### **Example Headlines**
- **Quick-test buttons** with truncated headlines
- **Random example generator** for variety
- **Thai examples**: รัฐบาล, ยารักษาโรค, etc.
- **English examples**: Government, medication, etc.

## 🛠 **Technical Implementation**

### **Architecture**
```
src/app.py                    # Main Streamlit application
├── Google-like CSS styling  # Custom CSS for modern UI
├── Bilingual support       # Thai/English language system
├── Model integration       # Offline predictor connection
└── Responsive design       # Mobile-friendly layout
```

### **Key Components**
1. **Custom CSS**: Google-inspired styling with animations
2. **Language System**: Dynamic text switching
3. **Model Integration**: Seamless connection to trained ML model
4. **State Management**: Streamlit session state for user experience
5. **Error Handling**: Graceful degradation and helpful messages

### **Styling Features**
- **Color Scheme**: Google's official colors (#4285f4, #ea4335, #34a853)
- **Typography**: Clean, readable fonts with proper hierarchy
- **Animations**: Smooth hover effects and loading states
- **Responsive**: Mobile-first design with breakpoints
- **Accessibility**: High contrast and proper ARIA labels

## 📊 **Demo Data**

### **Thai Examples (ตัวอย่างภาษาไทย)**
1. **Real News**: รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า
2. **Real News**: พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง
3. **Fake News**: วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์
4. **Fake News**: พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%

### **English Examples**
1. **Real News**: Government announces economic development plan
2. **Real News**: New effective diabetes medication discovered
3. **Fake News**: Scientists find miracle herb for weight loss in 1 week
4. **Fake News**: Coconut oil found to cure cancer with 100% effectiveness

## 🔧 **Customization**

### **Adding New Examples**
Edit the `examples` list in `src/app.py`:
```python
examples = [
    "Your new Thai headline here",
    "อีกหัวข้อข่าวใหม่",
    # Add more examples...
]
```

### **Changing Colors**
Modify the CSS color variables:
```css
.main-title .check { color: #4285f4; }  /* Google Blue */
.main-title .di { color: #ea4335; }     /* Google Red */
```

### **Adding Languages**
Extend the `LANGUAGES` dictionary:
```python
LANGUAGES = {
    'th': { ... },
    'en': { ... },
    'new_lang': {
        'title': 'CheckDi',
        'subtitle': 'Your translated subtitle',
        # Add all translations...
    }
}
```

## 🚨 **Troubleshooting**

### **Common Issues**

**❌ "Model not found" error**
```bash
# Solution: Train the model first
python train_offline.py
```

**❌ "Import error" for predictor**
```bash
# Solution: Check file structure
ls models/offline-thai-fakenews-classifier/
# Should contain: model.pkl, model_info.json, label_encoder.pkl
```

**❌ "Port already in use"**
```bash
# Solution: Use different port
streamlit run src/app.py --server.port 8502
```

**❌ Thai text not displaying**
```bash
# Solution: Install Thai fonts
# On macOS: Already included
# On Linux: sudo apt-get install fonts-thai-tlwg
# On Windows: Install Thai language support
```

## 📱 **Mobile Experience**

The demo is fully responsive and works great on mobile:

- **Adaptive layout** that stacks elements vertically
- **Touch-friendly buttons** with proper spacing
- **Readable text** at all screen sizes
- **Fast loading** with minimal JavaScript

## 🎉 **Demo Success Metrics**

Your CheckDi demo achieves:

- ✅ **Professional appearance** rivaling commercial applications
- ✅ **Bilingual support** for broader accessibility
- ✅ **Real ML integration** with actual trained model
- ✅ **Responsive design** working across all devices
- ✅ **User-friendly interface** requiring no technical knowledge
- ✅ **Google-like experience** that users immediately understand

## 🚀 **Next Steps**

After demoing CheckDi, consider:

1. **Deploy to cloud** (Streamlit Cloud, Heroku, AWS)
2. **Add more languages** (Chinese, Japanese, Vietnamese)
3. **Integrate larger models** (when computational resources allow)
4. **Add user authentication** for personalized features
5. **Implement usage analytics** to understand user behavior
6. **Create mobile app** using the same ML backend

---

**🎊 Your CheckDi demo is now ready to impress! 🎊**

Launch it with `python run_app.py` and show off your professional Thai fake news detection system!