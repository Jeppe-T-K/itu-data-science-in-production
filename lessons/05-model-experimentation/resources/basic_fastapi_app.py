import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Which model to serve: pick a registered model version (or a runs:/ URI)
MODEL_URI = os.environ.get("MODEL_URI", "models:/thresholded_logistic_model/1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per request
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(title="Thresholded classifier service", lifespan=lifespan)


class Invocation(BaseModel):
    # One or more rows of features, e.g. {"features": [[12.0, 34.0]]}
    features: list[list[float]]


class Prediction(BaseModel):
    predictions: list[int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/invocations")
def invocations(payload: Invocation) -> Prediction:
    # The threshold is inside the loaded model, not in this file
    df = pd.DataFrame(payload.features)
    return Prediction(predictions=[int(p) for p in app.state.model.predict(df)])
