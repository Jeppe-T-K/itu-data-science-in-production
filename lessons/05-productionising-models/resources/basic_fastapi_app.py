"""A possible solution for Exercise 3: serve the registered thresholded
classifier behind your own FastAPI app - model loaded once at startup, a
health endpoint that reports readiness, and a validated invocations
endpoint. Run with `uvicorn basic_fastapi_app:app --reload`."""

import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    # One or more rows of features (30 features per row for this model),
    # e.g. {"features": [[13.5, 21.7, ...], [20.6, 29.3, ...]]}
    features: list[list[float]]


class Prediction(BaseModel):
    # The wrapper's output: predictions plus the threshold that produced them
    predictions: list[int]
    threshold: float


@app.get("/health")
def health():
    # "ok" only once the model is actually loaded and ready to serve
    if getattr(app.state, "model", None) is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.post("/invocations")
def invocations(payload: Invocation) -> Prediction:
    # The threshold is inside the loaded model, not in this file
    df = pd.DataFrame(payload.features)
    predictions = app.state.model.predict(df)
    return Prediction(**predictions)
