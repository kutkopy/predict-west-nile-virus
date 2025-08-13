from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

app = FastAPI(
    title="West Nile Virus Prediction API",
    description="API for predicting West Nile Virus presence in mosquito traps",
    version="1.0.0"
)

model = None
label_encoders = None
feature_columns = None

@app.on_event("startup")
async def load_model():
    global model, label_encoders, feature_columns
    try:
        model = joblib.load('model/wnv_model.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')
        feature_columns = joblib.load('model/feature_columns.pkl')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class PredictionRequest(BaseModel):
    Date: str
    Species: str
    Latitude: float
    Longitude: float
    Tmax: Optional[float] = None
    Tmin: Optional[float] = None
    Tavg: Optional[float] = None
    Depart: Optional[float] = None
    DewPoint: Optional[float] = None
    WetBulb: Optional[float] = None
    Heat: Optional[float] = None
    Cool: Optional[float] = None
    PrecipTotal: Optional[float] = None
    StnPressure: Optional[float] = None
    SeaLevel: Optional[float] = None
    ResultSpeed: Optional[float] = None
    AvgSpeed: Optional[float] = None
    Sprayed: Optional[int] = 0

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str

def preprocess_input(data: PredictionRequest) -> pd.DataFrame:
    """Preprocess input data to match training format"""
    try:
        date = pd.to_datetime(data.Date)
        
        input_dict = {
            'Species': data.Species,
            'Latitude': data.Latitude,
            'Longitude': data.Longitude,
            'Tmax': data.Tmax,
            'Tmin': data.Tmin,
            'Tavg': data.Tavg,
            'Depart': data.Depart,
            'DewPoint': data.DewPoint,
            'WetBulb': data.WetBulb,
            'Heat': data.Heat,
            'Cool': data.Cool,
            'PrecipTotal': data.PrecipTotal,
            'StnPressure': data.StnPressure,
            'SeaLevel': data.SeaLevel,
            'ResultSpeed': data.ResultSpeed,
            'AvgSpeed': data.AvgSpeed,
            'Sprayed': data.Sprayed,
            'Year': date.year,
            'Month': date.month,
            'DayOfYear': date.dayofyear,
            'Week': date.isocalendar().week
        }
        
        df = pd.DataFrame([input_dict])
        
        if 'Species' in label_encoders:
            le = label_encoders['Species']
            species_value = data.Species
            if species_value in le.classes_:
                df['Species'] = le.transform([species_value])[0]
            else:
                df['Species'] = 0
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df.reindex(columns=feature_columns, fill_value=0)
        
        return df
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing input: {str(e)}")

@app.get("/")
async def root():
    return {"message": "West Nile Virus Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for West Nile Virus presence"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        processed_data = preprocess_input(request)
        
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0, 1]
        
        if probability < 0.3:
            confidence = "low"
        elif probability < 0.7:
            confidence = "medium"
        else:
            confidence = "high"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Make predictions for multiple samples"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        for request in requests:
            processed_data = preprocess_input(request)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0, 1]
            
            if probability < 0.3:
                confidence = "low"
            elif probability < 0.7:
                confidence = "medium"
            else:
                confidence = "high"
            
            results.append({
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": confidence
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": str(type(model)),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "label_encoders": list(label_encoders.keys()) if label_encoders else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)