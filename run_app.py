#!/usr/bin/env python3
"""
Launcher script for CheckDi Streamlit application
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    # Map package names to import names
    required_packages = {
        'streamlit': 'streamlit',
        'scikit-learn': 'sklearn', 
        'pandas': 'pandas',
        'numpy': 'numpy',
        'joblib': 'joblib'
    }
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_model():
    """Check if trained model exists"""
    model_path = "models/offline-thai-fakenews-classifier/model.pkl"
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        print("\n🏋️ Train the model first by running:")
        print("   python train_offline.py")
        print("\n📁 Expected model location:")
        print(f"   {model_path}")
        return False
    
    print("✅ Model found!")
    return True

def main():
    """Main launcher function"""
    print("🚀 CheckDi - Thai Fake News Detection App")
    print("=" * 50)
    
    # Check requirements
    print("📋 Checking requirements...")
    if not check_requirements():
        return
    
    print("✅ All packages installed!")
    
    # Check model
    print("\n🤖 Checking trained model...")
    if not check_model():
        return
    
    # Launch Streamlit app
    print("\n🌐 Launching Streamlit app...")
    print("📍 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8504")
    print("\n💡 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/app_final.py",
            "--server.port", "8504",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")

if __name__ == "__main__":
    main()