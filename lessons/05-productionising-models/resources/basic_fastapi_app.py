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
    # One or more rows of features, e.g. {"features": [[12.0, 34.0]]}
    features: list[list[float]]


class Prediction(BaseModel):
    # The wrapper's output: predictions plus a timestamp of the call
    predictions: list[int]
    timestamp: str


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
    return Prediction(**app.state.model.predict(df))
