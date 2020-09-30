import liwc_util
import re
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import spacy
from matplotlib import pyplot as plt
import sys
from scipy import stats
import vader_negation_util
import pprint
import argparse
import json
from pathlib import Path
import os

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")

def compute_negation_using_liwc(data, save_pickle_path, result, class_id, cluster_result, categories, category_reverse):
    
    data_filepath = data["data_filepath"]    
    required_categories = ["posemo", "negemo"]
    n_samples = None

    if "n_samples" in data:
        n_samples = data["n_samples"]
    
    review_level_liwc = []    
    with open(data_filepath, "r") as f:
        all_reviews = []
        for rev in f.readlines():
            rev = rev.strip("\n")
            all_reviews.append(rev)

        if n_samples == None:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=min(len(all_reviews),n_samples), replace=False)
        selected_reviews = [all_reviews[idx] for idx in indices]
        count = 0
        for rev in selected_reviews:
            count += 1
            doc = nlp(rev)            
            token_count = len(doc)
            review_level_count = {}
            sent_count = 0

            for sent in doc.sents:  
                sent_count += 1

            negation_pos = 0
            negation_neg = 0
            posemo_words = list()
            negemo_words = list()
            for idx,token in enumerate(doc):
                for cat in required_categories:
                    for pattern in cluster_result[category_reverse[cat]]:
                        if (pattern.endswith("*") and token.text.startswith(pattern[:-1])) or (pattern==token.text):
                            if cat == "posemo" and idx>0:
                                if doc[idx-1].text in vader_negation_util.NEGATE:
                                    negation_pos += 1
                                    posemo_words.append(doc[idx-1].text+" "+token.text)
                            elif cat == "negemo" and idx>0:
                                if doc[idx-1].text in vader_negation_util.NEGATE:
                                    negation_neg += 1
                                    negemo_words.append(doc[idx-1].text+" "+token.text)
                
            review_level_count["total_no_of_tokens"] = token_count
            review_level_count["total_no_of_sents"] = sent_count
            review_level_count["negation_posemo"] = negation_pos
            review_level_count["negation_negemo"] = negation_neg
            review_level_count["negemo_words"] = negemo_words
            review_level_count["posemo_words"] = posemo_words
            
            review_level_liwc.append(review_level_count)

        negation_pos_review_level = list(map(lambda x:x['negation_posemo'], review_level_liwc))
        negation_pos_sent_level = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_sents"], review_level_liwc))
        negation_pos_word_level = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_tokens"], review_level_liwc))

        negation_neg_review_level = list(map(lambda x:x['negation_negemo'], review_level_liwc))
        negation_neg_sent_level = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_sents"], review_level_liwc))
        negation_neg_word_level = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_tokens"], review_level_liwc))

        myprint(f"Negation dist. posemo, review-level: {np.mean(negation_pos_review_level)}")
        myprint(f"Negation dist. posemo, sent-level: {np.mean(negation_pos_sent_level)}")
        myprint(f"Negation dist. posemo, word-level: {np.mean(negation_pos_word_level)}")

        myprint(f"Negation dist. negemo, review-level: {np.mean(negation_neg_review_level)}")
        myprint(f"Negation dist. negemo, sent-level: {np.mean(negation_neg_sent_level)}")
        myprint(f"Negation dist. negemo, word-level: {np.mean(negation_neg_word_level)}")

        pickle.dump({
            "negation_analysis_data": review_level_liwc
        }, open(save_pickle_path, "wb"))
        


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--datasets_info_json",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,
                        help="")
    parser.add_argument("--liwc_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")   
    
    args = parser.parse_args()  
    myprint(f"args: {args}")     
    np.random.seed(args.seed_val)    
    result, class_id, cluster_result, categories, category_reverse = liwc_util.load_liwc(args.liwc_filepath)
    datasets = json.loads(open(args.datasets_info_json, "r").read())
    saves_dir = os.path.join(args.saves_dir_name, "negation_using_liwc")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    # plot_save_prefix = "liwc_negation"
    
    for data in datasets:               
        save_pickle_path = os.path.join(saves_dir, data["name"]+"_pos_reviews.pickle")
        compute_negation_using_liwc(data["positive"], save_pickle_path, result, class_id, 
            cluster_result, categories, category_reverse)
        save_pickle_path = os.path.join(saves_dir, data["name"]+"_neg_reviews.pickle")
        compute_negation_using_liwc(data["negative"], save_pickle_path, result, class_id, 
            cluster_result, categories, category_reverse)
        print()
        print()