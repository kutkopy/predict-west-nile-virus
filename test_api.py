import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Health check: {response.status_code} - {response.json()}")

def test_single_prediction():
    """Test single prediction"""
    test_data = {
        "Date": "2013-07-01",
        "Species": "CULEX PIPIENS",
        "Latitude": 41.954,
        "Longitude": -87.800,
        "Tmax": 85.0,
        "Tmin": 67.0,
        "Tavg": 76.0,
        "DewPoint": 65.0,
        "WetBulb": 70.0,
        "PrecipTotal": 0.0,
        "StnPressure": 29.85,
        "SeaLevel": 30.15,
        "ResultSpeed": 5.2,
        "AvgSpeed": 3.8,
        "Sprayed": 0
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=test_data)
    print(f"Single prediction: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"Error: {response.text}")

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{API_BASE_URL}/model/info")
    print(f"Model info: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"Features: {len(info['features'])}")
        print(f"Label encoders: {info['label_encoders']}")

if __name__ == "__main__":
    print("Testing West Nile Virus Prediction API...")
    print("=" * 50)
    
    try:
        print("1. Testing health endpoint...")
        test_health()
        print()
        
        print("2. Testing model info...")
        test_model_info()
        print()
        
        print("3. Testing single prediction...")
        test_single_prediction()
        print()
        
    except requests.exceptions.ConnectionError:
        print("API server is not running. Please start with: uvicorn api:app --reload")
    except Exception as e:
        print(f"Test error: {e}")