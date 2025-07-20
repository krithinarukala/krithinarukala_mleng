# score_headlines_api.py

import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from sentence_transformers import SentenceTransformer
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# initialize the app
app = FastAPI()

# try to load up the models
try:
    logging.info("Loading models...")
    clf = joblib.load("svm.joblib")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load models: {e}")
    raise

# define what we're expecting in the request body
class HeadlinesRequest(BaseModel):
    headlines: List[str]

# checking health of request
@app.get("/status")
async def status():
    logging.info("Health check requested.")
    return {"status": "OK"}

# scoring endpoint
@app.post("/score_headlines")
async def score_headlines(request: HeadlinesRequest):
    logging.info(f"Received request to score {len(request.headlines)} headlines.")

    if not request.headlines:
        logging.warning("Empty headline list received.")
        return {"labels": []}

    try:
        # Generate embeddings and predict
        embeddings = model.encode(request.headlines)
        labels = clf.predict(embeddings).tolist()
        logging.info("Returning predictions for headlines.")
        return {"labels": labels}

    except Exception as e:
        logging.error(f"Error during scoring: {e}")
        return {"error": "Internal server error"}

