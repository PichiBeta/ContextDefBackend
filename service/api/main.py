from fastapi import FastAPI
import os
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from difficulty_model.models.beto.train_beto import text_to_score
from pydantic import BaseModel, Field

ml_model = {}
REPO = "pichibeta/difficulty_prediction_SPA"
class PredictRequest(BaseModel):
    body: str = Field(..., min_length=1)
@asynccontextmanager
async def lifespan(app:FastAPI):
    token = os.getenv("HF_TOKEN")
    ml_model["tokenizer"] = AutoTokenizer.from_pretrained(REPO, token=token)
    ml_model["model"] = AutoModelForSequenceClassification.from_pretrained(REPO, token=token)
    yield
    ml_model.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(request: PredictRequest):
    result = text_to_score(ml_model["model"], ml_model["tokenizer"], request.body)
    return {"result": result}