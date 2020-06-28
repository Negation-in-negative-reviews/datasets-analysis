import os
import numpy as np

VADER_LEXICON_PATH = "/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"

def read_vader_sentiment_dict(filepath):
    vader_sentiment_scores = {}
    with open(filepath, "r") as fin:
        for line in fin:
            values = line.split("\t")
            vader_sentiment_scores[values[0]] = float(values[1])

    return vader_sentiment_scores

if __name__ == "__main__":
    vader_scores = read_vader_sentiment_dict(VADER_LEXICON_PATH)
    selected_lexicons = []
    for key, val in vader_scores.items():
        if val >= -1.5 and val < -1:
            selected_lexicons.append(key)

    print(selected_lexicons)
