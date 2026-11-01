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
    return Prediction(**app.state.model.predict(df))


# --- Tests (run with pytest) ---

from fastapi.testclient import TestClient


def sample_features():
    # Two real rows from the breast cancer dataset: the dataset is ordered
    # with the malignant samples first, so row 0 is malignant and row 500
    # is benign
    from sklearn.datasets import load_breast_cancer

    X, _ = load_breast_cancer(return_X_y=True)
    return X[[0, 500]].tolist()


def test_health_loading():
    # Without entering the lifespan the model is not loaded yet
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_ok():
    with TestClient(app) as client:  # enters the lifespan, loads the model
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_invocations():
    with TestClient(app) as client:
        response = client.post("/invocations", json={"features": sample_features()})
        assert response.status_code == 200
        assert set(response.json()) == {"predictions", "threshold"}
        # Known input -> expected class: malignant -> 0, benign -> 1
        assert response.json()["predictions"] == [0, 1]
