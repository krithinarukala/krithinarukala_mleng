# Krithin Arukala, PyForMLEng, Assignment 2
# the script scores the sentiment of headlines using
# a pre-trained SVM model and SentenceTransformer embeddings.
'''
This script takes a list of headlines via a POST request,
scores their sentiment using a pre-trained SVM model,
and returns the sentiment labels.
'''
import logging
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer

# configure logging settings
logging.basicConfig(
    # set the logging level and format
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# initialize the app
app = FastAPI()

# try to load up the models
try:
    logging.info("Loading the models...")
    # load the pre-trained SVM model and SentenceTransformer model
    clf = joblib.load("svm.joblib")
    # load the SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    # log the error if the model file is not found
    logging.critical("Model file not found: %s", e)
    raise
except Exception as e:
    # log any other exceptions that occur during model loading
    logging.critical("Failed to load model: %s", e)
    raise

# define what we're expecting in the request body
class HeadlinesRequest(BaseModel):
    """request model for headline scoring."""
    headlines: List[str]

# checking health of request
@app.get("/status")
async def status():
    """Health check endpoint."""
    # log the request
    logging.info("Received request to check service status.")
    return {"status": "OK"}

# scoring endpoint
@app.post("/score_headlines")
async def score_headlines(request: HeadlinesRequest):
    """Endpoint to score the sentiment of headlines."""
    # log the request
    logging.info("Received request to score the headlines.")

    if not request.headlines:
        # if the headlines list is empty, log a warning and return an empty response
        logging.warning("Received empty headlines list.")
        return {"labels": []}

    try:
        # generate embeddings and predict
        embeddings = model.encode(request.headlines)
        # complete the prediction
        labels = clf.predict(embeddings).tolist()
        logging.info("Returning predictions for headlines.")
        # return the labels
        return {"labels": labels}
    except Exception as e:
        # log the error and return a generic error message
        logging.error("Error during scoring: %s", e)
        return {"error": "Internal server error"}
