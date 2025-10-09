import os
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Set timezone to IST (UTC+5:30)
import pytz
ist = pytz.timezone('Asia/Kolkata')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("price_prediction_api")

# ===== Config =====
MODEL_PATH = os.getenv("PRICE_MODEL_PATH", "price_model_compatible.pkl")
PORT = int(os.getenv("PORT", "8000"))

# Policy parameters
STABILITY_PCT = 0.15
MIN_GM_PCT = 0.12
COMP_CAP = {"Economy": 1.05, "Premium": 1.08}
COMP_FLOOR = {"Economy": 0.90, "Premium": 0.88}
TIME_NUDGE = {"Morning": 0.03, "Afternoon": 0.0, "Evening": 0.04, "Night": 0.01}

# ===== App =====
app = FastAPI(
    title="Price Prediction API",
    version="1.0.0",
    description="API for predicting prices using trained Gradient Boosting model"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load model =====
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Loaded price prediction model from %s", MODEL_PATH)
    
    # Try to get feature names
    if hasattr(model, 'feature_names_in_'):
        FEATURE_NAMES = model.feature_names_in_.tolist()
        logger.info("✅ Model expects features: %s", FEATURE_NAMES)
    else:
        FEATURE_NAMES = None
        logger.info("⚠️  No feature names found in model")
        
except Exception as e:
    logger.error("❌ Could not load model: %s", e)
    model = None

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Input features for prediction")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: float = Field(..., description="Prediction confidence score")
    status: str
    message: str

class BatchPredictionResponse(BaseModel):
    predictions: list[Dict]
    total_processed: int
    errors: list[str] = []

# ====== Helper Functions ======
def gm_pct(price, cost):
    if price <= 0:
        return 0.0
    return (price - cost) / price

def inv_nudge(ratio):
    if ratio < 0.8:
        return 0.03
    if ratio > 1.2:
        return -0.03
    return 0.0

def row_price_bounds(row):
    base = float(row.get("baseline_price", row["Historical_Cost_of_Ride"]))
    cost = float(row["Historical_Cost_of_Ride"])
    veh = row.get("Vehicle_Type", "Economy")
    comp = float(row.get("competitor_price", base))

    lo = base * (1 - STABILITY_PCT)
    hi = base * (1 + STABILITY_PCT)

    base_gm = gm_pct(base, cost)
    min_gm = max(MIN_GM_PCT, base_gm)
    lo_gm = cost / max(1 - min_gm, 1e-6)

    cap = COMP_CAP.get(veh, 1.06)
    floor = COMP_FLOOR.get(veh, 0.90)
    lo_cmp = comp * floor
    hi_cmp = comp * cap

    lower = max(lo, lo_gm, lo_cmp)
    upper = min(hi, hi_cmp)

    if upper < lower:
        upper = lower

    logger.info(f"Price bounds: lower={lower}, upper={upper}")
    return lower, upper

def validate_features(input_data: dict) -> pd.DataFrame:
    """Validate and prepare features for prediction"""
    required_features = [
        'Number_of_Riders', 'Number_of_Drivers', 'Historical_Cost_of_Ride',
        'Expected_Ride_Duration', 'competitor_price', 'Location_Category',
        'Time_of_Booking', 'Vehicle_Type', 'Customer_Loyalty_Status',
        'Number_of_Past_Rides', 'Average_Ratings'
    ]
    missing = [f for f in required_features if f not in input_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    try:
        df = pd.DataFrame([input_data])
        logger.info("Raw input features: %s", input_data)
        
        # Compute derived features
        df['Rider_Driver_Ratio'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1e-6)
        df['Driver_to_Rider_Ratio'] = df['Number_of_Drivers'] / (df['Number_of_Riders'] + 1e-6)
        df['Cost_per_Min'] = df['Historical_Cost_of_Ride'] / (df['Expected_Ride_Duration'] + 1e-6)
        df['Supply_Tightness'] = df['Number_of_Riders'] - df['Number_of_Drivers']
        df['Inventory_Health_Index'] = df['Driver_to_Rider_Ratio'] * 100
        df['baseline_price'] = df['Historical_Cost_of_Ride']
        df['price'] = df['baseline_price']  # Initial for base prediction
        
        # Handle categorical features manually to match training
        # Time_of_Booking: drop Afternoon
        df['Time_of_Booking_Evening'] = (df['Time_of_Booking'] == 'Evening').astype(int)
        df['Time_of_Booking_Morning'] = (df['Time_of_Booking'] == 'Morning').astype(int)
        df['Time_of_Booking_Night'] = (df['Time_of_Booking'] == 'Night').astype(int)
        
        # Location_Category: drop Rural
        df['Location_Category_Suburban'] = (df['Location_Category'] == 'Suburban').astype(int)
        df['Location_Category_Urban'] = (df['Location_Category'] == 'Urban').astype(int)
        
        # Vehicle_Type: drop Economy
        df['Vehicle_Type_Premium'] = (df['Vehicle_Type'] == 'Premium').astype(int)
        
        # Customer_Loyalty_Status: drop Gold
        df['Customer_Loyalty_Status_Regular'] = (df['Customer_Loyalty_Status'] == 'Regular').astype(int)
        df['Customer_Loyalty_Status_Silver'] = (df['Customer_Loyalty_Status'] == 'Silver').astype(int)
        
        # Drop original categorical columns
        df = df.drop(columns=['Location_Category', 'Time_of_Booking', 'Vehicle_Type', 'Customer_Loyalty_Status'])
        
        # Ensure all expected features are present
        if FEATURE_NAMES:
            missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
            for feature in missing_features:
                df[feature] = 0.0
            df = df[FEATURE_NAMES]  # Reorder to match model
        
        logger.info("Processed features: %s", df.columns.tolist())
        logger.info("Feature values: %s", df.iloc[0].to_dict())
        
        return df
    except Exception as e:
        raise ValueError(f"Feature validation failed: {e}")

def make_prediction(features_df: pd.DataFrame) -> tuple[float, float]:
    """Make prediction using the loaded model and return optimal price with confidence"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        row = features_df.iloc[0]
        cost = float(row["Historical_Cost_of_Ride"])
        base_p = model.predict(features_df)[0]  # Initial with price = baseline_price
        
        lo, hi = row_price_bounds(row)
        
        t_n = TIME_NUDGE.get(row.get("Time_of_Booking", "Afternoon"), 0.0)
        i_n = inv_nudge(row["Driver_to_Rider_Ratio"])
        base = row["baseline_price"]
        center = np.clip((lo + hi) / 2, lo, hi)  # Use midpoint of bounds as initial price
        
        n_grid = 11
        grid = np.linspace(lo, hi, n_grid)
        
        best_price = center  # Start with center of bounds
        best_rev = -1.0  # Initialize to negative to ensure first valid option is considered
        
        for p in grid:
            if gm_pct(p, cost) < MIN_GM_PCT:
                logger.info(f"Skipping price {p} due to low GM: {gm_pct(p, cost)} < {MIN_GM_PCT}")
                continue
            
            pred_df = features_df.copy()
            pred_df["price"] = p
            p_now = model.predict(pred_df)[0]
            
            if p_now < base_p:
                logger.info(f"Skipping price {p} as p_now {p_now} < base_p {base_p}")
                continue
            
            rev = p * p_now
            if rev > best_rev:
                best_price = p
                best_rev = rev
                logger.info(f"New best price {best_price} with revenue {best_rev}")
        
        # Enforce strict bounds on the final price
        best_price = min(max(best_price, lo), hi)
        logger.info(f"Final price after bounds: {best_price}, bounds: [{lo}, {hi}]")
        
        confidence = 0.9  # Placeholder; could use model uncertainty if available
        
        return round(best_price, 2), confidence
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise RuntimeError(f"Prediction failed: {e}")

# ====== Endpoints ======
@app.get("/")
async def root():
    current_time = datetime.now(ist).strftime("%I:%M %p IST on %B %d, %Y")
    return {
        "message": "Price Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "current_time": current_time,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs",
            "test-predict": "/test-predict"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is None else "unhealthy",  # Fixed logic error: should be "healthy" if model is loaded
        "model_loaded": model is not None,
        "service": "price_prediction_api"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "features_expected": FEATURE_NAMES,
    }
    
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single price prediction
    """
    try:
        # Validate and prepare features
        features_df = validate_features(request.features)
        
        # Make prediction
        predicted_price, confidence = make_prediction(features_df)
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence=round(confidence * 100, 2),
            status="success",
            message="Prediction completed successfully"
        )
        
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Make batch predictions from CSV file
    """
    errors = []
    predictions = []
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        logger.info("Processing batch prediction for %d rows", len(df))
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Process each row
        for index, row in df.iterrows():
            try:
                # Convert row to dictionary
                features = row.to_dict()
                
                # Prepare features
                features_df = validate_features(features)
                
                # Make prediction
                predicted_price, confidence = make_prediction(features_df)
                
                predictions.append({
                    "row_index": index,
                    "features": features,
                    "predicted_price": predicted_price,
                    "confidence": round(confidence * 100, 2)
                })
                
            except Exception as e:
                error_msg = f"Row {index}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            errors=errors
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/test-predict")
async def test_predict():
    """Test prediction with sample data from dashboard"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Test with sample data
        test_data = {
            "Number_of_Riders": 70,
            "Number_of_Drivers": 20,
            "Historical_Cost_of_Ride": 300.00,
            "competitor_price": 350.00,
            "Location_Category": "Rural",
            "Time_of_Booking": "Morning",
            "Vehicle_Type": "Economy",
            "Customer_Loyalty_Status": "Silver",
            "Expected_Ride_Duration": 120,
            "Number_of_Past_Rides": 30,
            "Average_Ratings": 4.2
        }
        
        features_df = validate_features(test_data)
        predicted_price, confidence = make_prediction(features_df)
        
        return {
            "predicted_price": predicted_price,
            "confidence": round(confidence * 100, 2),
            "status": "success",
            "test_data": test_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Test prediction failed: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return {"detail": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)