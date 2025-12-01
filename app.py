from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "MJ ML API Working!"}

@app.post("/predict")
def predict(data: dict):
    value = data["input"]
    result = model.predict([[value]])[0]
    return {"prediction": int(result)}
