from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Cancer Prediction API"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
