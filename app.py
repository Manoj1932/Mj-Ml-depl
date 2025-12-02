from fastapi import FastAPI
from fastapi.responses import FileResponse
import joblib
import os

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)

@app.post("/predict")
def predict(data: dict):
    features = [
        data["feature1"],
        data["feature2"],
        data["feature3"]
    ]
    prediction = model.predict([features])[0]
    return {"prediction": prediction}
