---
title: Model Serving
---

# Exercise 3: Serving the Model

Your model now sits in the MLflow registry, threshold and all - but it isn't serving anyone yet. The last step from "model" to "product" is putting it behind an API, so other systems (or humans) can get predictions on demand.

There are two ways to do that, and you will try both:

1. **Serve the model directly with MLflow.** `mlflow models serve` starts a scoring server for any logged model, `/invocations` endpoint included. This is exactly how the course project serves its vial-classification model.
2. **Define the API yourself with FastAPI.** When you need control over the contract - validation, documentation, response bodies - you write the service yourself. The go-to Python framework for that is [FastAPI](https://fastapi.tiangolo.com/): it validates requests with [pydantic](https://docs.pydantic.dev/), generates interactive documentation automatically, and is easy to test. The course project uses both together: a thin FastAPI app in front of the MLflow scoring server.

### Exercise 3.1: Serve the model directly with MLflow

1. <details><summary>Start the scoring server</summary>

   <pre><code>mlflow models serve -m models:/thresholded_logistic_model/1 -p 5001 --no-conda</code></pre>

   Under the hood this launches a FastAPI application run by uvicorn - the same stack you will use yourself in a moment. The <code>-m</code> flag accepts any model URI: a registry version like here, or a <code>runs:/...</code> URI.
   </details>

2. <details><summary>Call the invocations endpoint</summary>

   <pre><code>curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{"instances": [[12.0, 34.0], [95.0, 60.0]]}'</code></pre>

   The scoring server converts the JSON into a DataFrame and passes it to your wrapper's <code>predict</code>. The response is your wrapper's output, wrapped by the scoring server in a <code>predictions</code> key:

   <pre><code>{"predictions": {"predictions": [0, 1], "timestamp": "2026-10-30T09:41:12.842017+00:00"}}</code></pre>

   The course project calls the same endpoint the same way - only with base64-encoded images in <code>instances</code> instead of feature rows - and lives with the same wrapping rather than writing its own scoring server.

   <pre><i><u>Where did the threshold come from in these predictions? What happens if you send the wrong field name, or a string instead of a number?</u></i></pre>
   </details>

3. <details><summary>Check the health endpoint</summary>

   <pre><code>curl -i http://localhost:5001/health</code></pre>

   <pre><i><u>Is the response valid JSON? This bare-newline answer is exactly why the course project puts its own FastAPI app in front of the scoring server: to control what the endpoints actually respond. Keep that motivation in mind for the next part.</u></i></pre>
   </details>

### Exercise 3.2 (advanced): Define your own API with FastAPI

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

   <pre><i><u>What is the /docs page? (hint: OpenAPI). Why does every serious service have a health endpoint - remember the logging/monitoring part of the lecture?</u></i></pre>
   </details>

2. <details><summary>Load the model once at startup</summary>
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

3. <details><summary>Add the invocations endpoint</summary>
   Define pydantic models for the request and response, so FastAPI both validates and documents the contract. The response mirrors the wrapper's output: predictions plus a timestamp.

   ```python
   from pydantic import BaseModel

   class Invocation(BaseModel):
       features: list[list[float]]

   class Prediction(BaseModel):
       predictions: list[int]
       timestamp: str

   @app.post("/invocations")
   def invocations(payload: Invocation) -> Prediction:
       df = pd.DataFrame(payload.features)
       return Prediction(**app.state.model.predict(df))
   ```

   Start the service and try it:

   <pre><code>curl -X POST http://localhost:8000/invocations \
     -H "Content-Type: application/json" \
     -d '{"features": [[12.0, 34.0], [95.0, 60.0]]}'</code></pre>

   <pre><i><u>Compare the response to the one from the scoring server in 3.1: here it is not double-wrapped - defining the API yourself means controlling the contract. What happens if you send a string instead of a number? Or the wrong field name? Watch what FastAPI does for you - and discuss why "sanity check input/output" from the lecture matters.</u></i></pre>
   </details>

4. <details><summary>Make the health check complete</summary>
   A health endpoint that just returns <code>{"status": "ok"}</code> only tells you the process is alive - not that it can actually serve. Make it report on the model as well: return <code>503</code> with a <code>"loading"</code> status until the model is loaded, and only then <code>200</code> with <code>"ok"</code>.

   The course project takes this further: its health check distinguishes <code>ok</code> / <code>loading</code> / <code>error</code>, because the model there runs in a separate process that can die independently of the API.

   <pre><i><u>Why does a load balancer (or an orchestrator like Docker/Kubernetes) care about the difference between "process alive" and "model ready"?</u></i></pre>
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

<pre><i><u>Discuss:

<li>The scoring server from 3.1 needed no code from you. When is that enough, and when do you need the FastAPI app from 3.2? What does the course project's combination of the two give it?</li>
<li>This is "on demand" serving. When would batch serving (like the one from the MLOps projects lesson) be better?</li>
<li>Who should be able to call your service? What is missing before you would call this production-ready? (hint: authentication, monitoring, containerization...)</li>
<li>The lecture said "recent != better" when promoting models. How does the MODEL_URI environment variable let you choose what gets served - and roll back?</li></u></i></pre>
