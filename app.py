from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    feature1 = float(data["feature1"])
    feature2 = float(data["feature2"])

    arr = np.array([[feature1, feature2]])
    pred = int(model.predict(arr)[0])

    return {"prediction": pred}
