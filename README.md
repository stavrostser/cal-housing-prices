# Housing Price Prediction Case Study

This repository contains a case study for housing price prediction using machine learning. The application is implemented in Python, leveraging PyTorch for model training and FastAPI for serving predictions via a REST API.  

## Features
- Custom preprocessing pipeline for robust data preparation
- Neural network model training and evaluation
- REST API for inference (with Swagger documentation)

---

## Table of Contents
1. [Requirements](#requirements)
2. [Setup Instructions](#setup-instructions)
3. [Software Architecture](#software-architecture)
4. [Architecture Diagram](#architecture-diagram)

---

## Requirements

- Python 3.12+ (needs to be installed)
- Libraries (can be installed via steps below):
  - `pandas`  
  - `numpy`  
  - `pytorch`  
  - `fastapi`  
  - `uvicorn`  
  - `matplotlib`  

---

## Setup Instructions

0. Clone this repository:
   ```bash
   git clone https://github.com/stavrostser/cal-housing-prices.git
   cd cal-housing-prices

1. (Optional) Create a virtual environment based on your OS / Editor 

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the inference API locally:
    ```bash
    uvicorn app:app --reload
    ```

4. Access the API documentation at: http://127.0.0.1:8000/docs


## Software Architecture
The project consists of three main components:

1. Data Pre-processing and Exploration:

- Read and pre-processe the dataset.
- Cleanup, handle missing values, outliers, feature engineering.
- Explore the dataset in a Jupyter notebook.

2. Model Training:

- Train a neural network model with PyTorch for regression.
- Evaluate the model and save to make available for inference.

3. Inference API:

- REST API built using FastAPI.
- Accepts JSON input and returns predicted housing price.
- API endpoint documentation available via Swagger UI.