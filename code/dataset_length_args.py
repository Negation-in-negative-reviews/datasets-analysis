import re
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import spacy
from matplotlib import pyplot as plt
import sys
from scipy import stats
import seaborn_plot_util
import pprint
import json
import os
from pathlib import Path
import util
import argparse


pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")

def compute_dataset_length(data_filepath, n_samples=None):
    all_reviews = []
    
    with open(data_filepath, "r") as fin:
        for line in fin:
            all_reviews.append(line.strip("\n"))

        if n_samples == None:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=n_samples)
        selected_reviews = [all_reviews[idx] for idx in indices]
        
        analysis_data = []
        for rev in selected_reviews:
            doc = nlp(rev)             
            token_count = len(doc)            
            sent_count = 0
            for sent in doc.sents:  
                sent_count += 1

            analysis_data.append({
                "token_count": token_count,
                "sent_count": sent_count,
                "review": rev
            })
        avg_no_of_tokens_per_review = np.mean(list(map(lambda x:x["token_count"], analysis_data)))
        avg_no_of_sents_per_review = np.mean(list(map(lambda x:x["sent_count"], analysis_data)))
        myprint(f"# of tokens per review: {avg_no_of_tokens_per_review}")
        myprint(f"# of sentences per review: {avg_no_of_sents_per_review}")
        return analysis_data

if __name__ == "__main__":  
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
    parser.add_argument("--name",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        help="")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    
    args = parser.parse_args()
    np.random.seed(args.seed_val)

    saves_dir = os.path.join(args.saves_dir_name, "dataset_length")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)       
    plot_data_sent_level = []
    plot_data_review_level = []
    plot_save_prefix = "dataset_length_dist"
    analysis_data = {}

    myprint(f"args: {args}")        

    analysis_data["positive"] = compute_dataset_length(args.pos_file, args.n_samples)       
    analysis_data["negative"] = compute_dataset_length(args.neg_file, args.n_samples)

    print()
    print()
    Path(os.path.join(saves_dir, "full", args.name)).mkdir(parents=True, exist_ok=True)
        
    pickle.dump({
        "analysis_data": analysis_data
    }, open(os.path.join(saves_dir, "full", args.name, "dataset_length_dist.pickle"), "wb"))    
    