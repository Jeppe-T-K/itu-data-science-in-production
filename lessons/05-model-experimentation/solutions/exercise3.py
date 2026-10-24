import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_URI = os.environ.get("MODEL_URI", "models:/thresholded_logistic_model/1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per request
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(title="Thresholded classifier service", lifespan=lifespan)


class Invocation(BaseModel):
    features: list[list[float]]


class Prediction(BaseModel):
    predictions: list[int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/invocations")
def invocations(payload: Invocation) -> Prediction:
    df = pd.DataFrame(payload.features)
    return Prediction(predictions=[int(p) for p in app.state.model.predict(df)])


# --- Tests (run with pytest) ---

from fastapi.testclient import TestClient


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invocations():
    with TestClient(app) as client:  # enters the lifespan, loads the model
        response = client.post(
            "/invocations",
            json={"features": [[12.0, 34.0], [95.0, 60.0]]},
        )
        assert response.status_code == 200
        assert set(response.json()) == {"predictions"}
        assert all(p in (0, 1) for p in response.json()["predictions"])
