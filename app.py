from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader, Dataset
import json
import pandas as pd
from ml.nn import NeuralNetworkRegressorAutoencoder, NeuralNetworkRegressor

# Load Models and Map
AUTOENCODER_PATH = "models/20241204-172803_modelAutoencoderLocation-revised.pt"
MODEL_PATH = "models/20241205-003506_model-revised.pt"
OCEAN_MAP_PATH = "dataset/ocean_proximity_category_mapping.json"

# Load the location autoencoder model
modelAutoencoderLocation = torch.load(AUTOENCODER_PATH, map_location=torch.device('cpu'))

# Load the main model
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Load ocean proximity mapping
with open(OCEAN_MAP_PATH, "r") as f:
    ocean_map = json.load(f)

# Function to map ocean_proximity to category
def map_ocean_proximity(ocean_value: str):
    if ocean_value not in ocean_map:
        raise ValueError(f"Invalid 'ocean_proximity': {ocean_value}")
    return ocean_map[ocean_value]

# Initialize FastAPI
app = FastAPI()

# Define Input Schema
class InputData(BaseModel):
    longitude: float = -122.25
    latitude: float = 37.85
    housing_median_age: float = 12
    households: int = 193
    median_income: float = 40368
    ocean_proximity: str = "NEAR BAY"

# API Endpoint for Inference
@app.post("/inference")
async def inference(data: InputData):
    try:
        
        # Map ocean_proximity to category
        ocean_proximity_category = map_ocean_proximity(data.ocean_proximity)

        # Input validation
        # NOTE: This is general validation. It should be restricted only to California for current models.
        if not (-180 <= data.longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180.")
        if not (-90 <= data.latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90.")
        if data.housing_median_age < 0 or data.households < 1 or data.median_income < 0:
            raise ValueError("Invalid values for housing_median_age, households, or median_income.")

        # Prepare location dataframe for autoencoder input
        autoencoder_input = pd.DataFrame([{
            "longitude": data.longitude,
            "latitude": data.latitude,
            "ocean_proximity_category": ocean_proximity_category
        }])

        # Pass through autoencoder
        autoencoder_output_df = modelAutoencoderLocation.predict(autoencoder_input)
        autoencoder_output = autoencoder_output_df["autoencoder_output"].iloc[0]

        # Create dataframe for main model prediction
        df = pd.DataFrame([{
            "housing_median_age": data.housing_median_age,
            "households": data.households,
            "median_income": data.median_income,
            "location": autoencoder_output,
            "median_house_value": 0  # initialize output
        }])

        # Make inference from the main model
        model.predict(df)

        pred = df["median_house_value"].iloc[0]
        
        return { "median_house_value": pred }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")