# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("iris_model.pkl")

# Create the FastAPI app
app = FastAPI()

# Define the data input model using Pydantic
class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the response model
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str

# API endpoint for making predictions
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    # Convert input data to numpy array for prediction
    input_data = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    # Make the prediction
    prediction = model.predict(input_data)
    class_name = ["setosa", "versicolor", "virginica"][prediction[0]]  # Convert numerical prediction to class name

    return PredictionResponse(prediction=prediction[0], class_name=class_name)

# Optionally, an endpoint to check if the API is working
@app.get("/")
def read_root():
    return {"message": "Machine learning model API is running!"}