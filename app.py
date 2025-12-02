from fastapi import FastAPI
from fastapi.responses import FileResponse
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Serve Frontend (predict.html)
@app.get("/")
def home():
    return FileResponse("predict.html")

# Predict API
@app.post("/predict")
def predict(feature1: float, feature2: float):
    data = np.array([[feature1, feature2]])
    result = model.predict(data)
    return {"prediction": int(result[0])}
