# Krithin Arukala, PyForMLEng, Assignment 2
# the script scores the sentiment of headlines using
# a pre-trained SVM model and SentenceTransformer embeddings.
'''
This script takes a list of headlines via a POST request,
scores their sentiment using a pre-trained SVM model,
and returns the sentiment labels.
'''
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from sentence_transformers import SentenceTransformer
import uvicorn

# configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# initialize the app
app = FastAPI()

# try to load up the models
try:
    logging.info("We are loading the models...")
    clf = joblib.load("svm.joblib")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("We have successfully loaded the models.")
except Exception as e:
    logging.critical(f"failed to load model: {e}")
    raise

# define what we're expecting in the request body
class HeadlinesRequest(BaseModel):
    headlines: List[str]

# checking health of request
@app.get("/status")
async def status():
    logging.info("Received request to check service status.")
    return {"status": "OK"}

# scoring endpoint
@app.post("/score_headlines")
async def score_headlines(request: HeadlinesRequest):
    logging.info(f"Received request to score the headlines.")

    if not request.headlines:
        logging.warning("Received empty headlines list.")
        return {"labels": []}

    try:
        # Generate embeddings and predict
        embeddings = model.encode(request.headlines)
        labels = clf.predict(embeddings).tolist()
        logging.info("Returning predictions for headlines.")
        return {"labels": labels}

    except Exception as e:
        logging.error(f"error during scoring: {e}")
        return {"error": "Internal server error"}
