# 🚀 CheckDi Quick Start Guide

## ✅ Everything is Ready!

Your CheckDi Thai fake news detection demo is fully set up and working!

## 🎯 Launch Your Demo

### **Simple Launch**
```bash
cd /Users/dhanabhon/Projects/Git/checkdi
python run_app.py
```

### **Direct Launch**
```bash
streamlit run src/app.py
```

## 🌐 **Access Your Demo**
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.1.107:8501 (accessible from other devices)

## 🎨 **What You'll See**

### **Google-Like Interface**
- Large **"CheckDi"** logo (blue + red Google colors)
- Clean search input with rounded corners
- Thai/English language toggle
- Example buttons for quick testing

### **Smart Predictions** 
- Real-time analysis of Thai news headlines
- Visual confidence bars (green for real, red for fake)
- Detailed probability percentages
- Technical model information

## 📝 **Try These Examples**

### **Thai Examples (ภาษาไทย)**
```
รัฐบาลเปิดเผยแผนพัฒนาเศรษฐกิจในปีหน้า
พบยารักษาโรคเบาหวานใหม่ที่มีประสิทธิภาพสูง
วิทยาศาสตร์ใหม่พบว่ากินใบย่านางช่วยลดน้ำหนักได้ภายใน 1 สัปดาห์
พบว่าน้ำมันมะพร้าวสามารถรักษาโรคมะเร็งได้ 100%
```

### **English Examples**
```
Government announces economic development plan for next year
New effective diabetes medication discovered by researchers
Scientists find miracle herb that helps lose weight in 1 week
Coconut oil found to cure cancer with 100% effectiveness
```

## 🎉 **Your Demo Features**

✅ **Professional UI** - Google-inspired design  
✅ **81.36% Accuracy** - Trained SVM model  
✅ **Bilingual Support** - Thai + English  
✅ **Real-time Analysis** - Instant predictions  
✅ **Mobile Responsive** - Works on all devices  
✅ **Example-driven** - Easy testing with sample headlines  

## 🔄 **Quick Test**

To verify everything works:
```bash
python -c "
import sys
sys.path.append('src')
from core.predictor_offline import test_predictor
test_predictor()
"
```

## 🛑 **Stop the Demo**
Press `Ctrl+C` in the terminal to stop the Streamlit app.

---

## 🎊 **You're All Set!**

Your CheckDi demo is professional, functional, and ready to impress!

**Launch command**: `python run_app.py`  
**Demo URL**: http://localhost:8501

Enjoy your Thai fake news detection system! 🔍✨