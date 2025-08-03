#!/usr/bin/env python3
"""
Setup Script for Used Car Price Prediction Web Application
==========================================================

This script sets up the complete machine learning web application
by training the model and starting the Flask server.

Usage:
    python setup.py

Features:
- Installs required packages
- Trains the ML model
- Tests the API
- Provides instructions for frontend usage
"""

import subprocess
import sys
import os
import time
import requests

def install_packages():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "xgboost",
        "flask",
        "flask-cors",
        "matplotlib",
        "seaborn",
        "requests"
    ]
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Failed to install {package}. Please install manually.")
    
    print("✅ Package installation completed!")

def train_model():
    """Train the machine learning model"""
    print("🤖 Training machine learning model...")
    
    try:
        result = subprocess.run([sys.executable, "model_training.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out!")
        return False
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def start_server():
    """Start the Flask server in background"""
    print("🚀 Starting Flask API server...")
    
    try:
        # Start Flask server in background
        server_process = subprocess.Popen(
            [sys.executable, "backend/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Test if server is running
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Flask API server started successfully!")
                print("🌐 Server running at: http://localhost:5000")
                return server_process
            else:
                print("❌ Server health check failed!")
                server_process.terminate()
                return None
        except requests.RequestException:
            print("❌ Could not connect to server!")
            server_process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def test_api():
    """Test the API endpoints"""
    print("🧪 Testing API endpoints...")
    
    try:
        # Test prediction
        test_data = {
            "brand": "Maruti Suzuki",
            "year": 2020,
            "km_driven": 50000,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "number_of_owners": "First"
        }
        
        response = requests.post(
            "http://localhost:5000/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API test successful!")
            print(f"   🚗 Test car: {test_data['brand']} {test_data['year']}")
            print(f"   💰 Predicted price: ₹{result['predicted_price']:.2f} lakhs")
            return True
        else:
            print("❌ API test failed!")
            return False
            
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def show_instructions():
    """Show usage instructions"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📋 How to use your Car Price Prediction App:")
    print("\n1. 🌐 Frontend (Web Interface):")
    print("   - Open 'frontend/index.html' in your web browser")
    print("   - Fill in car details in the form")
    print("   - Click 'Predict Price' to get predictions")
    
    print("\n2. 🔗 API Endpoints:")
    print("   - Health check: GET http://localhost:5000/health")
    print("   - Model info: GET http://localhost:5000/model-info") 
    print("   - Predict price: POST http://localhost:5000/predict")
    
    print("\n3. 📁 Project Files:")
    print("   - model_training.py: Train new models")
    print("   - backend/app.py: Flask API server")
    print("   - frontend/index.html: Web interface")
    print("   - models/model.pkl: Trained ML model")
    print("   - test_api.py: API testing script")
    
    print("\n4. 🛠️ Development:")
    print("   - Modify model_training.py to experiment with models")
    print("   - Update backend/app.py to add new API features")
    print("   - Customize frontend/index.html for better UI")
    
    print("\n⚠️ Important Notes:")
    print("   - Keep the Flask server running for API access")
    print("   - Model is trained on synthetic data for demonstration")
    print("   - Replace with real data for production use")
    
    print("\n🆘 Need help?")
    print("   - Run 'python test_api.py' to test API")
    print("   - Check Flask server logs for debugging")
    print("   - Retrain model: 'python model_training.py'")
    
    print("\n" + "=" * 60)
    print("🚀 Your Used Car Price Prediction App is ready!")
    print("=" * 60)

def main():
    """Main setup function"""
    print("🚗 Used Car Price Prediction - Setup Script")
    print("=" * 60)
    print("This script will set up your complete ML web application.")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('model_training.py'):
        print("❌ Please run this script from the project root directory!")
        print("   The directory should contain: model_training.py, backend/, frontend/")
        return
    
    # Step 1: Install packages
    try:
        install_packages()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user.")
        return
    
    # Step 2: Train model (if not exists or user wants to retrain)
    model_exists = os.path.exists('models/model.pkl')
    
    if not model_exists:
        print("\n📚 Model not found. Training new model...")
        if not train_model():
            print("❌ Setup failed during model training!")
            return
    else:
        print("\n📚 Existing model found.")
        retrain = input("   Do you want to retrain the model? (y/N): ").strip().lower()
        if retrain in ['y', 'yes']:
            if not train_model():
                print("❌ Setup failed during model training!")
                return
    
    # Step 3: Start server
    print("\n🚀 Starting application server...")
    server_process = start_server()
    
    if not server_process:
        print("❌ Setup failed during server startup!")
        return
    
    # Step 4: Test API
    time.sleep(2)  # Give server time to fully start
    if not test_api():
        print("⚠️ API test failed, but server is running.")
    
    # Step 5: Show instructions
    show_instructions()
    
    # Keep server running
    try:
        print(f"\n🔄 Flask server is running (PID: {server_process.pid})")
        print("Press Ctrl+C to stop the server and exit.")
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped. Goodbye!")

if __name__ == "__main__":
    main()
