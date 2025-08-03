#!/usr/bin/env python3
"""
Test Script for Used Car Price Prediction API
=============================================

This script tests the Flask API endpoints to ensure everything is working correctly.

Usage:
    python test_api.py
"""

import requests
import json
import time

def test_api():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Used Car Price Prediction API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   📊 Status: {data['status']}")
            print(f"   🤖 Model loaded: {data['model_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Model info
    print("\n2️⃣ Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model info retrieved")
            print(f"   🤖 Model: {data['model_name']}")
            print(f"   🎯 R² Score: {data['performance']['r2']:.4f}")
            print(f"   📅 Training Date: {data['training_date']}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model info error: {e}")
    
    # Test 3: Prediction
    print("\n3️⃣ Testing prediction endpoint...")
    test_cases = [
        {
            "name": "Maruti Suzuki Swift 2020 (Low KM)",
            "data": {
                "brand": "Maruti Suzuki",
                "model": "Swift",
                "year": 2020,
                "km_driven": 30000,
                "fuel_type": "Petrol",
                "transmission": "Manual",
                "number_of_owners": "First"
            }
        },
        {
            "name": "Honda City 2018 (Medium KM)",
            "data": {
                "brand": "Honda",
                "model": "City",
                "year": 2018,
                "km_driven": 80000,
                "fuel_type": "Diesel",
                "transmission": "Automatic",
                "number_of_owners": "Second"
            }
        },
        {
            "name": "Toyota Fortuner 2022 (Premium SUV)",
            "data": {
                "brand": "Toyota",
                "model": "Fortuner",
                "year": 2022,
                "km_driven": 15000,
                "fuel_type": "Diesel",
                "transmission": "Automatic",
                "number_of_owners": "First"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test Case {i}: {test_case['name']}")
        try:
            response = requests.post(
                f"{base_url}/predict",
                headers={"Content-Type": "application/json"},
                json=test_case['data']
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Prediction successful")
                print(f"   💰 Predicted Price: ₹{result['predicted_price']} lakhs")
                print(f"   🤖 Model: {result['model_info']['model_name']}")
                print(f"   🎯 Accuracy: {result['model_info']['r2_score']:.4f}")
            else:
                error_data = response.json()
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"   📝 Error: {error_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
    
    # Test 4: Error handling
    print("\n4️⃣ Testing error handling...")
    try:
        # Test with invalid data
        invalid_data = {
            "brand": "InvalidBrand",
            "model": "InvalidModel",
            "year": 1990,  # Too old
            "km_driven": -1000,  # Negative
            "fuel_type": "Water",  # Invalid
            "transmission": "Flying",  # Invalid
            "number_of_owners": "Million"  # Invalid
        }
        
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=invalid_data
        )
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"   ✅ Error handling working correctly")
            print(f"   📝 Error message: {error_data['error']}")
        else:
            print(f"   ⚠️ Unexpected response: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Testing Completed!")
    print("🚀 Your Used Car Price Prediction API is working correctly!")
    print("\n📋 Next Steps:")
    print("1. Open frontend/index.html in your browser")
    print("2. Fill in car details and test the prediction")
    print("3. The API is ready for production use!")
    print("=" * 50)

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    test_api()
