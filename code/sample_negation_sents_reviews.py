import spacy
import argparse
import vader_negation_util
import random
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
nlp = spacy.load("en_core_web_md")


def read_file(filename):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            line = line.strip("\n")
            reviews.append(line)
        return reviews

def is_negation(review):
    doc = nlp(review)
    neg_count = 0
    for sent in doc.sents:
        words = sent.text.strip("\n").split()
        neg_count += vader_negation_util.negated(words)

    if neg_count > 0:
        return True
    else:
        return False


def get_samples(data, n_samples):
    count = 0
    idx = 0
    sampled_neg_data = []
    while count < n_samples:
        sample = data[idx]
        if is_negation(sample):
            sampled_neg_data.append(sample)
            count += 1
        idx += 1
    return sampled_neg_data

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)
    random.seed(seed_val)

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--pos_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--neg_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    
    args = parser.parse_args()
    myprint(args)
    pos_data = read_file(args.pos_file)
    pos_negation_samples = get_samples(pos_data, n_samples=50)    
    print("Pos samples")
    for val in pos_negation_samples:
        print(val)
    neg_data = read_file(args.neg_file)
    neg_negation_samples = get_samples(neg_data, n_samples=50)
    print("\n\n\n")
    print("Neg samples")
    for val in neg_negation_samples:
        print(val)
