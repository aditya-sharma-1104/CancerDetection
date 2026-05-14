from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_cancer

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    cancer_type: str
    features: list

@app.get("/")
def home():
    return {"message": "Cancer Detection API Running"}

@app.post("/predict")
def predict(data: InputData):

    result = predict_cancer(
        data.cancer_type,
        data.features
    )

    return result
