# Krithin Arukala, PyForMLEng, Assignment 1
# script scores the sentiment of headlines using a pre-trained SVM model and SentenceTransformer embeddings.
'''
This script takes an input file containing headlines and a source identifier,
then processes the headlines to generate sentiment scores using a pre-trained SVM model.
It saves the sentiment scores along with the headlines to an output file.
'''
import argparse
import datetime
import joblib
import sys
from sentence_transformers import SentenceTransformer
import numpy as np

def main(input_file, source):
    '''
    Main function to process the input file, generate embeddings, and predict sentiment scores.
    Args:
        input_file (str): Path to the input text file containing headlines.
        source (str): Source identifier for the headlines, e.g., 'nyt', 'chicagotribune', etc.
    '''
    # load the trained SVM model and the SentenceTransformer model
    clf = joblib.load('svm.joblib')
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # loading the input file and source
    with open(input_file, 'r', encoding='utf-8') as file:
        headlines = [line.strip() for line in file.readlines()]

    # convert to embeddings
    embeddings = model.encode(headlines)

    # save the embeddings to a file
    embeddings_filename = f"embeddings_{source}_{datetime.datetime.today().strftime('%Y_%m_%d')}.npy"
    np.save(embeddings_filename, embeddings)
    print(f"embeddings are saved to {embeddings_filename}")

    # predict the sentiment scores using the SVM model
    predictions = clf.predict(embeddings)

    # prepare the output filename
    today = datetime.datetime.today().strftime('%Y_%m_%d')
    output_filename = f"headline_scores_{source}_{today}.txt"

    # write predictions and headlines to the output file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for headline, prediction in zip(headlines, predictions):
            output_file.write(f"{prediction}, {headline}\n")
    print(f"the predictions and headlines are saved to {output_filename}")

if __name__ == "__main__":
    # arg parse for command line arguments
    parser = argparse.ArgumentParser(description="identify the sentiment of headlines with an SVM model")

    # input file and source arguments
    parser.add_argument("input_file", help="this is the path to the input text file with the headlines")
    parser.add_argument("source", help="this is the source of the headlines, such as 'nyt', 'chicagotribune', 'latimes', etc.")

    # parsing the arguments
    args = parser.parse_args()

    # check if input file exists
    if not args.input_file or not args.source:
        print("Issue: 'input_file' and 'source' arguments are required. Please ensure the format is as follows: python score_headlines.py <input_file> <source>")
        sys.exit(1)

    # call the main function with parsed arguments
    main(args.input_file, args.source)
