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
import seaborn_plot_util_old
import json
from pathlib import Path
import os
import csv
import util
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")
# VADER_LEXICON_PATH = "/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


def read_vader_sentiment_dict(filepath):
    vader_sentiment_scores = {}
    with open(filepath, "r") as fin:
        for line in fin:
            values = line.split("\t")
            vader_sentiment_scores[values[0]] = float(values[1])
    return vader_sentiment_scores

def compute_vadersentiment(data, dataset_name, vader_sentiment_scores):
    
    data_filepath = data["data_filepath"]
    # saves_dir = os.path.join("saves", dataset_name)
    # Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    # save_pickle_path = os.path.join(saves_dir, "vader_pos_neg_dist_full_analysis_data.pickle")

    n_samples = None
    if "n_samples" in data:
        n_samples = data["n_samples"]
    
    review_data = []    
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
            pos_words = []
            neg_words = []
            matched_tokens_count = 0

            for sent in doc.sents:  
                sent_count += 1

            for idx,token in enumerate(doc):
                if token.text in vader_sentiment_scores:
                    matched_tokens_count+=1
                    sent_score = vader_sentiment_scores[token.text]
                    if sent_score>=1:
                        pos_words.append(token.text)
                    elif sent_score<=-1:
                        neg_words.append(token.text)

            review_level_count["review_text"] = rev
            review_level_count["total_no_of_tokens"] = token_count
            review_level_count["total_no_of_sents"] = sent_count                        
            review_level_count["neg_words"] = neg_words
            review_level_count["pos_words"] = pos_words
            review_level_count["neg_words_count"] = len(neg_words)
            review_level_count["pos_words_count"] = len(pos_words)
            review_level_count["matched_tokens_count"] = matched_tokens_count
            
            review_data.append(review_level_count)

        pos_review_level = list(map(lambda x:x['pos_words_count'], review_data))
        pos_sent_level = list(map(lambda x:1.0*x['pos_words_count']/x["total_no_of_sents"], review_data))
        pos_word_level = list(map(lambda x:1.0*x['pos_words_count']/x["total_no_of_tokens"], review_data))

        neg_review_level = list(map(lambda x:x['neg_words_count'], review_data))        
        neg_sent_level = list(map(lambda x:1.0*x['neg_words_count']/x["total_no_of_sents"], review_data))        
        neg_word_level = list(map(lambda x:1.0*x['neg_words_count']/x["total_no_of_tokens"], review_data))        

        myprint(f"Positive words count, review-level: {np.mean(pos_review_level)}")
        myprint(f"Positive words count, sent-level: {np.mean(pos_sent_level)}")
        myprint(f"Positive words count, word-level: {np.mean(pos_word_level)}")

        myprint(f"Negative words count, review-level: {np.mean(neg_review_level)}")
        myprint(f"Negative words count, sent-level: {np.mean(neg_sent_level)}")
        myprint(f"Negative words count, word-level: {np.mean(neg_word_level)}")

        # pickle.dump({
        #     "negation_analysis_data": review_data
        # }, open(save_pickle_path, "wb"))

        return review_data

def compute_vadersentiment_util(data, name, vader_sentiment_scores, category,
    plot_data, analysis_types):

    analysis_data = compute_vadersentiment(data, name, vader_sentiment_scores)

    for analysis in analysis_types:
        if analysis == "review_level":   
            pos_count_normalized = list(map(lambda x:x['pos_words_count'], analysis_data))        
        elif analysis == "sent_level":  
            pos_count_normalized = list(map(lambda x:1.0*x['pos_words_count']/x["total_no_of_sents"], analysis_data))       
        else:
            pos_count_normalized = list(map(lambda x:1.0*x['pos_words_count']/x["total_no_of_tokens"], analysis_data))        

        if analysis == "review_level":   
            neg_count_normalized = list(map(lambda x:x['neg_words_count'], analysis_data))        
        elif analysis == "sent_level":  
            neg_count_normalized = list(map(lambda x:1.0*x['neg_words_count']/x["total_no_of_sents"], analysis_data))       
        else:
            neg_count_normalized = list(map(lambda x:1.0*x['neg_words_count']/x["total_no_of_tokens"], analysis_data))
          
           
        plot_data[analysis].append({
            "category": "positive - "+category+" review ",
            "review category": category,
            "text sentiment": "positive",
            "name": name,
            "value": np.mean(pos_count_normalized),
            "sem_value": stats.sem(pos_count_normalized),
            "all_samples_data": pos_count_normalized
        })
        plot_data[analysis].append({
            "category": "negative - "+category+" review ",
            "review category": category,
            "text sentiment": "negative",
            "name": name,
            "value": np.mean(neg_count_normalized),
            "sem_value": stats.sem(neg_count_normalized),
            "all_samples_data": neg_count_normalized
        })


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
                        # required=True,
                        help="")   
    parser.add_argument("--vader_lexicon_path",
                        default=None,
                        type=str,
                        required=True,
                        help="")   
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    
    args = parser.parse_args()    
    np.random.seed(args.seed_val)
    myprint(f"args: {args}")

    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }

    saves_dir = os.path.join(args.saves_dir_name, "vader_pos_neg_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "vader_pos_neg_dist"

    analysis_types = list(plot_data.keys())
    
    vader_sentiment_scores = read_vader_sentiment_dict(args.vader_lexicon_path)
    datasets = json.loads(open(args.datasets_info_json).read())
    
    for data in datasets:
        myprint(data)                
        compute_vadersentiment_util(data["positive"], data["name"], vader_sentiment_scores, "positive", 
            plot_data, analysis_types)
        
        compute_vadersentiment_util(data["negative"], data["name"], vader_sentiment_scores, "negative", 
            plot_data, analysis_types)
        print()
        print()

    pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))