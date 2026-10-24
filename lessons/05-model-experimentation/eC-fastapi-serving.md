---
title: FastAPI Serving
---

# Exercise 3: Serving with FastAPI

Your model now sits in the MLflow registry, threshold and all — but it isn't serving anyone yet. The last step from "model" to "product" is putting it behind an API, so other systems (or humans) can get predictions on demand.

We will use [FastAPI](https://fastapi.tiangolo.com/), the go-to Python framework for model services: it validates requests with [pydantic](https://docs.pydantic.dev/), generates interactive documentation automatically, and is easy to test. The service you build here — a `/health` endpoint plus an `/invocations` endpoint — is exactly the shape you will meet again in the course project.

### Exercise 3.1: A minimal service

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

   Run it with <code>uvicorn app:app --reload</code> and open http://localhost:8000/health — then http://localhost:8000/docs.

   <pre><i><u>What is the /docs page? (hint: OpenAPI). Why does every serious service have a health endpoint — remember the logging/monitoring part of the lecture?</u></i></pre>
   </details>

### Exercise 3.2: Serve the model

1. <details><summary>Load the model once at startup</summary>
   Use a lifespan handler to load your registered wrapper once when the app starts:

   ```python
   from contextlib import asynccontextmanager

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       app.state.model = mlflow.pyfunc.load_model(
           os.environ.get("MODEL_URI", "models:/thresholded_logistic_model/1")
       )
       yield

   app = FastAPI(lifespan=lifespan)
   ```

   <pre><i><u>Why load at startup instead of inside the endpoint? What would that do to latency? And why read the model URI from an environment variable?</u></i></pre>
   </details>

2. <details><summary>Add the invocations endpoint</summary>
   Define pydantic models for the request and response, so FastAPI both validates and documents the contract:

   ```python
   from pydantic import BaseModel

   class Invocation(BaseModel):
       features: list[list[float]]

   class Prediction(BaseModel):
       predictions: list[int]

   @app.post("/invocations")
   def invocations(payload: Invocation) -> Prediction:
       df = pd.DataFrame(payload.features)
       return Prediction(predictions=[int(p) for p in app.state.model.predict(df)])
   ```

   Start the service and try it:

   <pre><code>curl -X POST http://localhost:8000/invocations \
     -H "Content-Type: application/json" \
     -d '{"features": [[12.0, 34.0], [95.0, 60.0]]}'</code></pre>

   <pre><i><u>What happens if you send a string instead of a number? Or the wrong field name? Watch what FastAPI does for you — and discuss why "sanity check input/output" from the lecture matters.</u></i></pre>
   </details>

3. <details><summary>Check the whole contract</summary>
   Send samples with a probability close to your threshold so that the output flips between classes.

   <pre><i><u>Where did the threshold come from in this prediction? (It should come from inside the logged model — if you find yourself re-specifying it in the service, revisit Exercise 2.)</u></i></pre>
   </details>

### Exercise 3.3: Test the service

1. <details><summary>Write an API test</summary>
   FastAPI ships with a test client, so you can test the service without running a server:

   ```python
   from fastapi.testclient import TestClient

   def test_health():
       response = TestClient(app).get("/health")
       assert response.status_code == 200
       assert response.json() == {"status": "ok"}
   ```

   Add tests for <code>/invocations</code> too (e.g. known input &rarr; expected class).
   </details>

2. <details><summary>Discuss: wire it into CI</summary>
   <pre><i><u>You set up GitHub Actions in the previous lesson. What would a workflow step that runs these tests give you that the unit tests of the model alone do not?</u></i></pre>
   </details>

### Exercise 3.4: Reflect

<pre><i><u>Discuss in pairs:

<li>This is "on demand" serving. When would batch serving (like the one from the MLOps projects lesson) be better?</li>
<li>Who should be able to call your service? What is missing before you would call this production-ready? (hint: authentication, monitoring, containerization...)</li>
<li>The lecture said "recent != better" when promoting models. How does the MODEL_URI environment variable let you choose what gets served — and roll back?</li></u></i></pre>

[Next: Exercise 4 - Check Performance of Deployed Model](model-monitoring)
