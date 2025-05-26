from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

@app.get("/")
def read_root():
    return "PU-Track Model Server"

model = joblib.load("./model.pkl")

class PredictionInput(BaseModel):
    avg_air_temp: float
    avg_air_humid: float
    cushion_slope: float
    elapsed_time: float

@app.post("/predict")
def predict(data: PredictionInput):
    # 입력 데이터를 모델 형식에 맞게 리스트로 변환
    X_new = [[
        data.avg_air_temp,
        data.avg_air_humid,
        data.cushion_slope,
        data.elapsed_time
    ]]
    # 예측
    prediction = model.predict(X_new)

    return {"predicted_remaining_time": prediction[0]}