from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/predict", response_class=HTMLResponse)
async def predict_page():
    with open("static/predict.html") as f:
        return HTMLResponse(f.read())


@app.post("/predict")
async def predict(request: Request):
    data = await request.form()
    try:
        f1 = float(data["feature1"])
        f2 = float(data["feature2"])
        f3 = float(data["feature3"])
        f4 = float(data["feature4"])
    except:
        return {"error": "Invalid input"}

    x = np.array([[f1, f2, f3, f4]])
    y_pred = model.predict(x)[0]

    return {"prediction": int(y_pred)}
