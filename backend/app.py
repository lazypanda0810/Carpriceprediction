#!/usr/bin/env python3
"""
Used Car Price Prediction Flask API
===================================

Flask backend API for used car price prediction.
Loads the trained ML model and provides REST endpoints.

Usage:
    python app.py

Endpoints:
    GET  /          - API information
    GET  /health    - Health check  
    GET  /model-info - Model information
    POST /predict   - Predict car price

Author: AI Assistant
Date: August 2025
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')

# Global variable to store model data
model_data = None

def load_model():
    """
    Load the trained model and preprocessing components
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model_data
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found at: {MODEL_PATH}")
            print("Please run model_training.py first to train and save the model!")
            return False
            
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model: {model_data['model_name']}")
        print(f"🎯 R² Score: {model_data['performance']['r2']:.4f}")
        print(f"📅 Training Date: {model_data['training_date']}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        traceback.print_exc()
        return False

def predict_car_price(brand, model, year, km_driven, fuel_type, transmission, number_of_owners, location='Mumbai'):
    """
    Predict car price based on input features
    
    Args:
        brand (str): Car brand
        model (str): Car model
        year (int): Year of manufacture
        km_driven (int): Kilometers driven
        fuel_type (str): Fuel type
        transmission (str): Transmission type
        number_of_owners (str): Number of owners
        location (str): Location (optional)
        
    Returns:
        float: Predicted price in lakhs
        
    Raises:
        ValueError: If model not loaded or invalid inputs
    """
    if model_data is None:
        raise ValueError("Model not loaded")
    
    try:
        # Encode categorical variables
        encoded_features = {}
        
        # Handle categorical encoding with fallback for unknown categories
        categorical_inputs = {
            'brand': brand,
            'model': model,
            'fuel_type': fuel_type, 
            'transmission': transmission,
            'number_of_owners': number_of_owners,
            'location': location
        }
        
        for col, value in categorical_inputs.items():
            encoder = model_data['encoders'][col]
            if value in encoder.classes_:
                encoded_features[f'{col}_encoded'] = encoder.transform([value])[0]
            else:
                # Use the most common category as fallback
                encoded_features[f'{col}_encoded'] = 0
                print(f"⚠️ Unknown {col}: {value}, using default encoding")
        
        # Feature engineering (same as in training)
        car_age = 2024 - year
        km_per_year = km_driven / (car_age + 1) if car_age > 0 else km_driven
        is_luxury_brand = 1 if brand in ['Honda', 'Toyota'] else 0
        high_mileage = 1 if km_driven > 100000 else 0
        
        # Create feature array in the same order as training
        features = [
            year,
            km_driven,
            car_age,
            km_per_year,
            is_luxury_brand,
            high_mileage,
            encoded_features['brand_encoded'],
            encoded_features['model_encoded'],
            encoded_features['fuel_type_encoded'],
            encoded_features['transmission_encoded'],
            encoded_features['number_of_owners_encoded'],
            encoded_features['location_encoded']
        ]
        
        features_array = np.array([features])
        
        # Make prediction
        if model_data['model_name'] == 'Linear Regression':
            features_scaled = model_data['scaler'].transform(features_array)
            prediction = model_data['model'].predict(features_scaled)[0]
        else:
            prediction = model_data['model'].predict(features_array)[0]
        
        # Ensure minimum price
        prediction = max(0.5, prediction)
        
        return round(prediction, 2)
        
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Used Car Price Prediction API',
        'version': '1.0.0',
        'status': 'active',
        'model_loaded': model_data is not None,
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/model-info': 'GET - Get model information',
            '/predict': 'POST - Predict car price'
        },
        'usage': {
            'predict_endpoint': {
                'method': 'POST',
                'content_type': 'application/json',
                'required_fields': [
                    'brand', 'model', 'year', 'km_driven', 'fuel_type', 
                    'transmission', 'number_of_owners'
                ],
                'optional_fields': ['location']
            }
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'timestamp': datetime.now().isoformat(),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model_data is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please run model_training.py first to train and save the model'
        }), 500
    
    return jsonify({
        'model_name': model_data['model_name'],
        'performance': model_data['performance'],
        'training_date': model_data['training_date'],
        'feature_count': len(model_data['feature_cols']),
        'features': model_data['feature_cols'],
        'dataset_info': model_data['dataset_info'],
        'category_mappings': model_data['category_mappings']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict car price based on input features
    
    Expected JSON input:
    {
        "brand": "Maruti Suzuki",
        "model": "Swift",
        "year": 2020,
        "km_driven": 50000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "number_of_owners": "First",
        "location": "Mumbai"  // optional
    }
    
    Returns:
    {
        "predicted_price": 5.25,
        "price_range": "₹5.25 lakhs", 
        "input_details": {...},
        "model_info": {...}
    }
    """
    try:
        if model_data is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please run model_training.py first to train and save the model'
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Please send car details as JSON in the request body'
            }), 400
        
        # Validate required fields
        required_fields = ['brand', 'model', 'year', 'km_driven', 'fuel_type', 'transmission', 'number_of_owners']
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields,
                'provided_fields': list(data.keys())
            }), 400
        
        # Extract and validate input values
        brand = str(data['brand']).strip()
        model = str(data['model']).strip()
        year = int(data['year'])
        km_driven = int(data['km_driven'])
        fuel_type = str(data['fuel_type']).strip()
        transmission = str(data['transmission']).strip()
        number_of_owners = str(data['number_of_owners']).strip()
        location = str(data.get('location', 'Mumbai')).strip()
        
        # Validate ranges
        if not (2000 <= year <= 2024):
            return jsonify({'error': 'Year must be between 2000-2024'}), 400
        if not (0 <= km_driven <= 500000):
            return jsonify({'error': 'KM driven must be between 0-500,000'}), 400
        if km_driven < 0:
            return jsonify({'error': 'KM driven cannot be negative'}), 400
        
        # Validate categorical values
        valid_brands = list(model_data['category_mappings']['brand'].values())
        valid_models = list(model_data['category_mappings']['model'].values())
        valid_fuel_types = list(model_data['category_mappings']['fuel_type'].values())
        valid_transmissions = list(model_data['category_mappings']['transmission'].values())
        valid_owners = list(model_data['category_mappings']['number_of_owners'].values())
        valid_locations = list(model_data['category_mappings']['location'].values())
        
        if brand not in valid_brands:
            return jsonify({
                'error': f'Invalid brand: {brand}',
                'valid_brands': valid_brands
            }), 400
        
        if model not in valid_models:
            return jsonify({
                'error': f'Invalid model: {model}',
                'valid_models': valid_models
            }), 400
        
        if fuel_type not in valid_fuel_types:
            return jsonify({
                'error': f'Invalid fuel type: {fuel_type}',
                'valid_fuel_types': valid_fuel_types
            }), 400
        
        if transmission not in valid_transmissions:
            return jsonify({
                'error': f'Invalid transmission: {transmission}',
                'valid_transmissions': valid_transmissions
            }), 400
        
        if number_of_owners not in valid_owners:
            return jsonify({
                'error': f'Invalid number of owners: {number_of_owners}',
                'valid_owners': valid_owners
            }), 400
        
        # Make prediction
        predicted_price = predict_car_price(
            brand, model, year, km_driven, fuel_type, 
            transmission, number_of_owners, location
        )
        
        # Calculate car age for response
        car_age = 2024 - year
        
        response = {
            'predicted_price': float(predicted_price),
            'price_range': f'₹{predicted_price} lakhs',
            'currency': 'INR (lakhs)',
            'input_details': {
                'brand': brand,
                'model': model,
                'year': int(year),
                'car_age': int(car_age),
                'km_driven': int(km_driven),
                'fuel_type': fuel_type,
                'transmission': transmission,
                'number_of_owners': number_of_owners,
                'location': location
            },
            'model_info': {
                'model_name': model_data['model_name'],
                'r2_score': float(model_data['performance']['r2']),
                'mae': float(model_data['performance']['mae']),
                'rmse': float(model_data['performance']['rmse'])
            },
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/health', '/model-info', '/predict']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'Please check the HTTP method for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def main():
    """Main function to start the Flask application"""
    print("🚗 Used Car Price Prediction API")
    print("=" * 50)
    
    # Load the model
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"📊 Model: {model_data['model_name']}")
        print(f"🎯 R² Score: {model_data['performance']['r2']:.4f}")
        print(f"📅 Training Date: {model_data['training_date']}")
    else:
        print("⚠️  Model not loaded. Some endpoints may not work.")
        print("📝 Please run model_training.py first!")
    
    print("=" * 50)
    print("🌐 API Endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  GET  /model-info - Model information")
    print("  POST /predict   - Predict car price")
    print("=" * 50)
    print("🔗 Server starting at: http://localhost:5000")
    print("🌍 Frontend should connect to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

if __name__ == '__main__':
    main()
    app.run(debug=True, host='0.0.0.0', port=5000)
