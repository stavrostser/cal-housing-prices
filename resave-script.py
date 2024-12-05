
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from ml.nn import CustomDataset, CustomDatasetAutoencoder, SimpleNN, NeuralNetworkRegressor, SimpleNNAutoencoder, NeuralNetworkRegressorAutoencoder

# Load Models and Map
AUTOENCODER_PATH = "models/20241204-172803_modelAutoencoderLocation.pt"
MODEL_PATH = "models/20241205-003506_model.pt"
OCEAN_MAP_PATH = "dataset/ocean_proximity_category_mapping.json"

# Load and Re-save the models 

# Load the location autoencoder model
modelAutoencoderLocation = torch.load(AUTOENCODER_PATH, map_location=torch.device('cpu'))

torch.save(modelAutoencoderLocation, "models/20241204-172803_modelAutoencoderLocation-revised.pt")

# Load the main model
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

torch.save(model, "models/20241205-003506_model-revised.pt")