
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ------------------------------------------------------------
# CREATE FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="Car Price Prediction API",
    description="Predict car price using a Linear Regression model",
    version="1.0"
)


# ------------------------------------------------------------
# INPUT DATA FORMAT
# ------------------------------------------------------------

class CarData(BaseModel):

    enginesize: float
    carwidth: float
    highwaympg: float
    boreratio: float
    carheight: float
    peakrpm: float


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Car Price Prediction API is running!"
    }


# ------------------------------------------------------------
# PREDICTION API
# ------------------------------------------------------------

@app.post("/predict")
def predict_price(car: CarData):

    # Convert input into dictionary
    input_data = {
        "enginesize": car.enginesize,
        "carwidth": car.carwidth,
        "highwaympg": car.highwaympg,
        "boreratio": car.boreratio,
        "carheight": car.carheight,
        "peakrpm": car.peakrpm
    }

    # Create DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    # Scale input using the SAME scaler used during training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Return prediction
    return {
        "predicted_price": round(float(prediction[0]), 2)
    }
