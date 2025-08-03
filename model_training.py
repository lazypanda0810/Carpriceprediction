#!/usr/bin/env python3
"""
Used Car Price Prediction Model Training Script
===============================================

This script trains machine learning models to predict used car prices
in the Indian market using synthetic data and saves the best model.

Usage:
    python model_training.py

Output:
    - model.pkl: Trained model with preprocessing components
    - training_report.txt: Model performance report
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime
import os

warnings.filterwarnings('ignore')

def create_synthetic_dataset(n_samples=5000):
    """
    Create a synthetic dataset for used car price prediction
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic car dataset
    """
    print("🔄 Generating synthetic used car dataset...")
    
    np.random.seed(42)
    
    # Define categories
    brands = ['Maruti Suzuki', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 
              'Toyota', 'Ford', 'Renault', 'Nissan', 'Chevrolet']
    
    # Car models by brand
    models_by_brand = {
        'Maruti Suzuki': ['Swift', 'Baleno', 'Alto', 'Wagon R', 'Dzire', 'Vitara Brezza', 'Ertiga', 'Ciaz'],
        'Hyundai': ['Creta', 'i20', 'Verna', 'Venue', 'Elantra', 'Tucson', 'i10', 'Santro'],
        'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'CR-V', 'Civic', 'Accord', 'BR-V'],
        'Tata': ['Nexon', 'Harrier', 'Tiago', 'Tigor', 'Altroz', 'Safari', 'Punch', 'Hexa'],
        'Mahindra': ['XUV500', 'Scorpio', 'Bolero', 'XUV300', 'Thar', 'KUV100', 'Marazzo', 'XUV700'],
        'Toyota': ['Innova', 'Fortuner', 'Etios', 'Yaris', 'Camry', 'Corolla', 'Glanza', 'Urban Cruiser'],
        'Ford': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle', 'Mustang', 'Fiesta', 'Ikon'],
        'Renault': ['Duster', 'Kwid', 'Captur', 'Lodgy', 'Fluence', 'Pulse', 'Scala', 'Triber'],
        'Nissan': ['Magnite', 'Kicks', 'Terrano', 'Micra', 'Sunny', 'X-Trail', 'Evalia', 'GT-R'],
        'Chevrolet': ['Beat', 'Spark', 'Cruze', 'Trailblazer', 'Tavera', 'Aveo', 'Captiva', 'Enjoy']
    }
    
    fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
    transmissions = ['Manual', 'Automatic']
    owner_types = ['First', 'Second', 'Third', 'Fourth & Above']
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 
                 'Pune', 'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow']
    
    # Generate features
    selected_brands = np.random.choice(brands, n_samples)
    selected_models = []
    
    # Generate corresponding models for each brand
    for brand in selected_brands:
        model = np.random.choice(models_by_brand[brand])
        selected_models.append(model)
    
    data = {
        'brand': selected_brands,
        'model': selected_models,
        'year': np.random.randint(2008, 2024, n_samples),
        'km_driven': np.random.randint(5000, 200000, n_samples),
        'fuel_type': np.random.choice(fuel_types, n_samples, 
                                    p=[0.4, 0.35, 0.1, 0.1, 0.05]),
        'transmission': np.random.choice(transmissions, n_samples, p=[0.7, 0.3]),
        'number_of_owners': np.random.choice(owner_types, n_samples, 
                                           p=[0.6, 0.25, 0.1, 0.05]),
        'location': np.random.choice(locations, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic selling prices based on features
    base_price = 5.0  # Base price in lakhs
    
    # Price multipliers for different features
    brand_multipliers = {
        'Maruti Suzuki': 1.0, 'Hyundai': 1.1, 'Honda': 1.2, 'Tata': 0.9,
        'Mahindra': 0.95, 'Toyota': 1.3, 'Ford': 1.1, 'Renault': 0.85,
        'Nissan': 1.0, 'Chevrolet': 0.8
    }
    
    # Model multipliers (premium models cost more)
    model_multipliers = {
        # Maruti Suzuki
        'Swift': 1.1, 'Baleno': 1.15, 'Alto': 0.8, 'Wagon R': 0.85, 'Dzire': 1.0, 
        'Vitara Brezza': 1.2, 'Ertiga': 1.05, 'Ciaz': 1.1,
        # Hyundai
        'Creta': 1.3, 'i20': 1.1, 'Verna': 1.2, 'Venue': 1.15, 'Elantra': 1.4, 
        'Tucson': 1.5, 'i10': 0.9, 'Santro': 0.85,
        # Honda
        'City': 1.2, 'Amaze': 1.0, 'Jazz': 1.1, 'WR-V': 1.15, 'CR-V': 1.6, 
        'Civic': 1.4, 'Accord': 1.8, 'BR-V': 1.25,
        # Tata
        'Nexon': 1.1, 'Harrier': 1.3, 'Tiago': 0.9, 'Tigor': 0.95, 'Altroz': 1.05, 
        'Safari': 1.4, 'Punch': 1.0, 'Hexa': 1.2,
        # Mahindra
        'XUV500': 1.2, 'Scorpio': 1.15, 'Bolero': 1.0, 'XUV300': 1.1, 'Thar': 1.3, 
        'KUV100': 0.9, 'Marazzo': 1.05, 'XUV700': 1.4,
        # Toyota
        'Innova': 1.4, 'Fortuner': 1.6, 'Etios': 1.0, 'Yaris': 1.2, 'Camry': 1.8, 
        'Corolla': 1.3, 'Glanza': 1.0, 'Urban Cruiser': 1.1,
        # Ford
        'EcoSport': 1.1, 'Endeavour': 1.4, 'Figo': 0.9, 'Aspire': 1.0, 'Freestyle': 1.05, 
        'Mustang': 2.0, 'Fiesta': 1.1, 'Ikon': 0.8,
        # Renault
        'Duster': 1.1, 'Kwid': 0.8, 'Captur': 1.2, 'Lodgy': 1.0, 'Fluence': 1.15, 
        'Pulse': 0.9, 'Scala': 0.95, 'Triber': 1.0,
        # Nissan
        'Magnite': 1.1, 'Kicks': 1.2, 'Terrano': 1.15, 'Micra': 0.9, 'Sunny': 1.0, 
        'X-Trail': 1.4, 'Evalia': 1.05, 'GT-R': 3.0,
        # Chevrolet
        'Beat': 0.9, 'Spark': 0.85, 'Cruze': 1.2, 'Trailblazer': 1.3, 'Tavera': 1.1, 
        'Aveo': 0.95, 'Captiva': 1.25, 'Enjoy': 1.0
    }
    
    fuel_multipliers = {
        'Petrol': 1.0, 'Diesel': 1.1, 'CNG': 0.9, 'LPG': 0.85, 'Electric': 1.4
    }
    
    transmission_multipliers = {'Manual': 1.0, 'Automatic': 1.15}
    
    owner_multipliers = {
        'First': 1.0, 'Second': 0.85, 'Third': 0.7, 'Fourth & Above': 0.6
    }
    
    location_multipliers = {
        'Mumbai': 1.2, 'Delhi': 1.15, 'Bangalore': 1.1, 'Chennai': 1.05,
        'Kolkata': 1.0, 'Pune': 1.08, 'Hyderabad': 1.06, 'Ahmedabad': 1.03,
        'Jaipur': 0.95, 'Lucknow': 0.9
    }
    
    selling_prices = []
    for _, row in df.iterrows():
        price = base_price
        
        # Apply multipliers
        price *= brand_multipliers[row['brand']]
        price *= model_multipliers.get(row['model'], 1.0)  # Use 1.0 if model not found
        price *= fuel_multipliers[row['fuel_type']]
        price *= transmission_multipliers[row['transmission']]
        price *= owner_multipliers[row['number_of_owners']]
        price *= location_multipliers[row['location']]
        
        # Year effect (depreciation)
        age = 2024 - row['year']
        price *= max(0.3, 1 - (age * 0.08))
        
        # KM driven effect
        km_factor = max(0.3, 1 - (row['km_driven'] / 300000))
        price *= km_factor
        
        # Add random noise
        price *= np.random.normal(1.0, 0.1)
        
        # Ensure minimum price
        price = max(0.5, price)
        
        selling_prices.append(round(price, 2))
    
    df['selling_price'] = selling_prices
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"📊 Price range: ₹{df['selling_price'].min():.2f} - ₹{df['selling_price'].max():.2f} lakhs")
    
    return df

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: Processed features and target, along with encoders
    """
    print("🔄 Preprocessing data...")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Remove outliers using IQR method
    Q1 = df_processed['selling_price'].quantile(0.25)
    Q3 = df_processed['selling_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_size = len(df_processed)
    df_processed = df_processed[
        (df_processed['selling_price'] >= lower_bound) & 
        (df_processed['selling_price'] <= upper_bound)
    ]
    print(f"📉 Removed {original_size - len(df_processed)} outliers")
    
    # Feature engineering
    df_processed['car_age'] = 2024 - df_processed['year']
    df_processed['km_per_year'] = df_processed['km_driven'] / (df_processed['car_age'] + 1)
    
    # Create luxury brand indicator
    luxury_brands = ['Honda', 'Toyota']
    df_processed['is_luxury_brand'] = df_processed['brand'].isin(luxury_brands).astype(int)
    
    # Create high mileage indicator
    df_processed['high_mileage'] = (df_processed['km_driven'] > 100000).astype(int)
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'number_of_owners', 'location']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
        encoders[col] = le
    
    # Select features for modeling
    feature_cols = [
        'year', 'km_driven', 'car_age', 'km_per_year', 'is_luxury_brand', 'high_mileage',
        'brand_encoded', 'model_encoded', 'fuel_type_encoded', 'transmission_encoded', 
        'number_of_owners_encoded', 'location_encoded'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['selling_price']
    
    print(f"✅ Preprocessing completed")
    print(f"📊 Features: {len(feature_cols)}")
    print(f"📊 Samples after preprocessing: {len(X)}")
    
    return X, y, encoders, feature_cols

def train_models(X, y):
    """
    Train multiple machine learning models
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        dict: Dictionary containing trained models and their performance
    """
    print("🔄 Training machine learning models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=15,
            min_samples_split=5
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1
        )
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use scaled data for Linear Regression, original for tree-based models
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"✅ {name} Results:")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")
    
    # Find best model based on R² score
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model_info = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_model_info['r2']:.4f}")
    print(f"   RMSE: {best_model_info['rmse']:.4f}")
    print(f"   MAE: {best_model_info['mae']:.4f}")
    
    return results, best_model_name, scaler

def save_model(best_model_name, results, encoders, feature_cols, scaler, dataset_info):
    """
    Save the best model and all preprocessing components
    
    Args:
        best_model_name (str): Name of the best performing model
        results (dict): Model training results
        encoders (dict): Label encoders for categorical variables
        feature_cols (list): List of feature column names
        scaler (StandardScaler): Fitted scaler for features
        dataset_info (dict): Information about the dataset
    """
    print("🔄 Saving model and preprocessing components...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Prepare model data for saving
    model_data = {
        'model': results[best_model_name]['model'],
        'model_name': best_model_name,
        'scaler': scaler,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'performance': {
            'mae': results[best_model_name]['mae'],
            'rmse': results[best_model_name]['rmse'],
            'r2': results[best_model_name]['r2']
        },
        'dataset_info': dataset_info,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'category_mappings': {
            encoder_name: {
                i: label for i, label in enumerate(encoder.classes_)
            } for encoder_name, encoder in encoders.items()
        }
    }
    
    # Save the model
    model_path = os.path.join(models_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to: {model_path}")
    
    # Create training report
    report_path = os.path.join(models_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("Used Car Price Prediction - Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {model_data['training_date']}\n")
        f.write(f"Dataset Size: {dataset_info['total_samples']}\n")
        f.write(f"Features Used: {len(feature_cols)}\n\n")
        
        f.write("Model Performance Comparison:\n")
        f.write("-" * 30 + "\n")
        for model_name, model_results in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  MAE:  {model_results['mae']:.4f}\n")
            f.write(f"  RMSE: {model_results['rmse']:.4f}\n")
            f.write(f"  R²:   {model_results['r2']:.4f}\n\n")
        
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Best R² Score: {results[best_model_name]['r2']:.4f}\n\n")
        
        f.write("Features:\n")
        for i, feature in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {feature}\n")
        
        f.write("\nCategory Mappings:\n")
        for encoder_name, mapping in model_data['category_mappings'].items():
            f.write(f"\n{encoder_name}:\n")
            for code, label in mapping.items():
                f.write(f"  {code}: {label}\n")
    
    print(f"📊 Training report saved to: {report_path}")
    
    return model_path

def test_saved_model(model_path):
    """
    Test the saved model with sample predictions
    
    Args:
        model_path (str): Path to the saved model
    """
    print("🔄 Testing saved model...")
    
    # Load the saved model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model: {model_data['model_name']}")
    print(f"🎯 R² Score: {model_data['performance']['r2']:.4f}")
    
    # Test with sample data
    test_cases = [
        {
            'brand': 'Maruti Suzuki',
            'model': 'Swift',
            'year': 2020,
            'km_driven': 50000,
            'fuel_type': 'Petrol',
            'transmission': 'Manual',
            'number_of_owners': 'First',
            'location': 'Mumbai'
        },
        {
            'brand': 'Honda',
            'model': 'City',
            'year': 2018,
            'km_driven': 80000,
            'fuel_type': 'Diesel',
            'transmission': 'Automatic',
            'number_of_owners': 'Second',
            'location': 'Delhi'
        }
    ]
    
    print("\n🔮 Sample Predictions:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        # Encode categorical variables
        encoded_features = {}
        for col, encoder in model_data['encoders'].items():
            if test_case[col] in encoder.classes_:
                encoded_features[f'{col}_encoded'] = encoder.transform([test_case[col]])[0]
            else:
                encoded_features[f'{col}_encoded'] = 0  # Default for unknown categories
        
        # Feature engineering
        car_age = 2024 - test_case['year']
        km_per_year = test_case['km_driven'] / (car_age + 1)
        is_luxury_brand = 1 if test_case['brand'] in ['Honda', 'Toyota'] else 0
        high_mileage = 1 if test_case['km_driven'] > 100000 else 0
        
        # Create feature array
        features = [
            test_case['year'],
            test_case['km_driven'],
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
        
        print(f"Test Case {i}:")
        print(f"  Car: {test_case['brand']} {test_case['model']} ({test_case['year']})")
        print(f"  KM: {test_case['km_driven']:,}, Fuel: {test_case['fuel_type']}")
        print(f"  Transmission: {test_case['transmission']}, Owner: {test_case['number_of_owners']}")
        print(f"  🏷️  Predicted Price: ₹{prediction:.2f} lakhs")
        print()

def main():
    """
    Main function to execute the complete model training pipeline
    """
    print("🚗 Used Car Price Prediction - Model Training")
    print("=" * 60)
    
    # Step 1: Create synthetic dataset
    df = create_synthetic_dataset(n_samples=5000)
    
    # Save dataset for reference
    dataset_path = os.path.join(os.getcwd(), 'dataset.csv')
    df.to_csv(dataset_path, index=False)
    print(f"💾 Dataset saved to: {dataset_path}")
    
    # Step 2: Preprocess data
    X, y, encoders, feature_cols = preprocess_data(df)
    
    dataset_info = {
        'total_samples': len(df),
        'features_count': len(feature_cols),
        'price_range': {
            'min': df['selling_price'].min(),
            'max': df['selling_price'].max(),
            'mean': df['selling_price'].mean()
        }
    }
    
    # Step 3: Train models
    results, best_model_name, scaler = train_models(X, y)
    
    # Step 4: Save the best model
    model_path = save_model(best_model_name, results, encoders, feature_cols, scaler, dataset_info)
    
    # Step 5: Test the saved model
    test_saved_model(model_path)
    
    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Model saved at: {model_path}")
    print(f"📊 Best Model: {best_model_name}")
    print(f"🎯 R² Score: {results[best_model_name]['r2']:.4f}")
    print("🚀 Ready for deployment!")
    print("=" * 60)

if __name__ == "__main__":
    main()
