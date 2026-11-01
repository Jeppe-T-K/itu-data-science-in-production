---
title: Model Serving
---

# Exercise 3: Serving the Model

Your model now sits in the MLflow registry, threshold and all - but it isn't serving anyone yet. The last step from "model" to "product" is putting it behind an API, so other systems (or humans) can get predictions on demand.

There are two ways to do that, and you will try both:

1. **Serve the model directly with MLflow.** `mlflow models serve` starts a scoring server for any logged model, `/invocations` endpoint included. This is exactly how the course project serves its vial-classification model.
2. **Define the API yourself with FastAPI.** When you need control over the contract - validation, documentation, response bodies - you can write the service yourself. A simple Python framework for that is [FastAPI](https://fastapi.tiangolo.com/): it validates requests with [pydantic](https://docs.pydantic.dev/), generates interactive documentation automatically, and is easy to test. The course project uses both together: a thin FastAPI app in front of the MLflow scoring server.

### Exercise 3.1: Serve the model directly with MLflow

1. <details><summary>Start the scoring server</summary>

   ```bash
   mlflow models serve -m models:/thresholded_logistic_model/1 -p 5001 --env-manager local
   ```

   Under the hood this launches a FastAPI application run by uvicorn. You specify which model to serve with the <code>-m</code> flag, which accepts any model URI: a registry version like here, or a <code>runs:/...</code> URI. 

   The <code>--env-manager local</code> flag tells MLflow to serve the model in your current Python environment. Without it, MLflow first builds a fresh virtual environment from the model's logged requirements - safer, but slow, and often the first thing that breaks on a laptop.
   </details>

2. <details><summary>Call the invocations endpoint</summary>
   Your model expects rows of 30 features - real measurements of cell nuclei, not numbers you would want to type by hand. Two example rows (a malignant sample and a benign one) are ready in <a href="https://github.com/Jeppe-T-K/itu-data-science-in-production/blob/main/lessons/05-productionising-models/resources/payload.json">resources/payload.json</a>, and <a href="https://github.com/Jeppe-T-K/itu-data-science-in-production/blob/main/lessons/05-productionising-models/resources/payload.csv">resources/payload.csv</a> shows the same two rows with their column names. Download the JSON and call the endpoint with it:

   ```bash
   curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d @payload.json
   ```

   The scoring server converts the JSON into a DataFrame and passes it to your wrapper's <code>predict</code>. The response is your wrapper's output, wrapped by the scoring server in a <code>predictions</code> key:

   ```json
   {"predictions": {"predictions": [0, 1], "threshold": 0.25}}
   ```

   <i><u>Where did the threshold in the response come from? What happens if you send the wrong field name, or a string instead of a number?</u></i>
   </details>

3. <details><summary>Check the health endpoint</summary>

   ```bash
   curl -i http://localhost:5001/health
   ```

   <i><u>Is the response valid JSON? This bare-newline answer is exactly why the course project puts its own FastAPI app in front of the scoring server: to control what the endpoints actually respond. Keep that motivation in mind for the next part.</u></i>
   </details>

### Exercise 3.2: Define your own API with FastAPI

1. <details><summary>Install and run a hello-world app</summary>
   <code>pip install fastapi uvicorn</code>

   Create <code>app.py</code>:

   ```python
   from fastapi import FastAPI

   app = FastAPI(title="Model service")

   @app.get("/health")
   def health():
       return {"status": "ok"}
   ```

   Run it with <code>uvicorn app:app --reload</code> and open http://localhost:8000/health - then http://localhost:8000/docs.

   <i><u>What is the /docs page? (hint: OpenAPI). Why does a health endpoint make sense to have? Think back on the logging/monitoring part of the lecture</u></i>
   </details>

2. <details><summary>The skeleton of the model service</summary>
   It's not a learning objective to set up a fully functioning API service. However, we want to give enough context for what it takes to serve a model via an API. Here is therefore a skeleton of the full service. Replace your <code>app.py</code> with it and read through: it loads your registered model when the app starts, defines the request and response contract with pydantic, and answers health checks. The <code>invocations</code> endpoint is yours to finish in the next step.

   ```python
   import os
   from contextlib import asynccontextmanager

   import mlflow
   import pandas as pd
   from fastapi import FastAPI
   from fastapi.responses import JSONResponse
   from pydantic import BaseModel

   MODEL_URI = os.environ.get("MODEL_URI", "models:/thresholded_logistic_model/1")


   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # Load the model once at startup, not per request
       app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
       yield


   app = FastAPI(title="Thresholded classifier service", lifespan=lifespan)


   class Invocation(BaseModel):
       # One or more rows of features (30 features per row for this model)
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
       ...  # your job in the next step
   ```

   </details>

3. <details><summary>Make the invocations endpoint work</summary>
   The endpoint body is only two lines: create a DataFrame from the request features, and feed it to the loaded model for the response.

   ```python
   @app.post("/invocations")
   def invocations(payload: Invocation) -> Prediction:
       df = pd.DataFrame(<payload operation>)
       predictions = <app.state.model prediction operations>
       return Prediction(<predictions output format>)
   ```

   `payload` is an `Invocation` item, so you need to get the `features` values out somehow.

   Start the service with <code>uvicorn app:app --reload</code> and try it - reuse the payload from 3.1, only with the other field name (FastAPI expects <code>features</code> where the scoring server expected <code>instances</code>):

   ```bash
   python -c "import json; d=json.load(open('payload.json')); json.dump({'features': d['instances']}, open('payload_fastapi.json','w'))"

   curl -X POST http://localhost:8000/invocations \
     -H "Content-Type: application/json" \
     -d @payload_fastapi.json
   ```

   <i><u>Compare the response to the one from the scoring server in 3.1: here it is not double-wrapped - defining the API yourself means controlling the contract. What happens if you send a string instead of a number? Or the wrong field name? Watch what FastAPI does for you - and discuss why "sanity check input/output" from the lecture matters.</u></i>

   <details><summary>Solution hints</summary>

   - For `<payload operation>`: the request rows are in <code>payload.features</code> - pydantic has already parsed and validated the JSON for you.
   - For `<app.state.model prediction operations>`: <code>app.state.model.predict(df)</code> returns a dict with <code>predictions</code> and <code>threshold</code>.
   - For `<predictions output format>`: unpack that dict into the response model with <code>**</code>, so <code>Prediction(**predictions)</code>.
   </details>
   </details>
