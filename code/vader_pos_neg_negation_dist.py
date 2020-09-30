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
import seaborn_plot_util_old
import pprint
import json
from pathlib import Path
import os
import util
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")
# VADER_LEXICON_PATH = "/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"
def read_vader_sentiment_dict(filepath):
    vader_sentiment_scores = {}
    with open(filepath, "r") as fin:
        for line in fin:
            values = line.split("\t")
            vader_sentiment_scores[values[0]] = float(values[1])

    return vader_sentiment_scores

def compute_vadersentiment(data, dataset_name, vader_sentiment_scores, saves_dir):
    
    data_filepath = data["data_filepath"]
    # saves_dir = os.path.join(, dataset_name)
    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    save_pickle_path = os.path.join(saves_dir, dataset_name+"_vader_pos_neg_negation_dist.pickle")

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

        for rev in selected_reviews:
            doc = nlp(rev)            
            token_count = len(doc)
            review_level_count = {}
            sent_count = 0

            for sent in doc.sents:  
                sent_count += 1

            negation_pos_count = 0
            negation_neg_count = 0
            posemo_words = list()
            negemo_words = list()
            matched_tokens_count = 0

            for idx,token in enumerate(doc):
                if token.text in vader_sentiment_scores:
                    matched_tokens_count += 1
                    sent_score = vader_sentiment_scores[token.text]
                    if sent_score>=1 and idx>0:
                        if doc[idx-1].text in vader_negation_util.NEGATE or "n't" in doc[idx-1].text:
                            negation_pos_count += 1
                            posemo_words.append(doc[idx-1].text+' '+token.text)
                    elif sent_score<=-1 and idx>0:
                        if doc[idx-1].text in vader_negation_util.NEGATE or "n't" in doc[idx-1].text:
                            negation_neg_count += 1
                            negemo_words.append(doc[idx-1].text+' '+token.text)

            review_level_count["total_no_of_tokens"] = token_count
            review_level_count["total_no_of_sents"] = sent_count
            review_level_count["negation_posemo"] = negation_pos_count
            review_level_count["negation_negemo"] = negation_neg_count
            review_level_count["negative_negation_words"] = negemo_words
            review_level_count["positive_negation_words"] = posemo_words
            review_level_count["matched_tokens_count"] = matched_tokens_count
            
            review_data.append(review_level_count)

        negation_pos_review_level = list(map(lambda x:x['negation_posemo'], review_data))
        negation_pos_sent_level = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_sents"], review_data))
        negation_pos_word_level = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_tokens"], review_data))

        negation_neg_review_level = list(map(lambda x:x['negation_negemo'], review_data))
        negation_neg_sent_level = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_sents"], review_data))
        negation_neg_word_level = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_tokens"], review_data))

        myprint(f"Negation dist. posemo, review-level: {np.mean(negation_pos_review_level)}")
        myprint(f"Negation dist. posemo, sent-level: {np.mean(negation_pos_sent_level)}")
        myprint(f"Negation dist. posemo, word-level: {np.mean(negation_pos_word_level)}")

        myprint(f"Negation dist. negemo, review-level: {np.mean(negation_neg_review_level)}")
        myprint(f"Negation dist. negemo, sent-level: {np.mean(negation_neg_sent_level)}")
        myprint(f"Negation dist. negemo, word-level: {np.mean(negation_neg_word_level)}")

        # myprint(f"Negative words: {list(map(lambda x: x['negative_words'], review_data))}")
        # myprint(f"Positive words: {list(map(lambda x: x['positive_words'], review_data))}")

        pickle.dump({
            "negation_analysis_data": review_data
        }, open(save_pickle_path, "wb"))
        
        return review_data

def compute_vadersentiment_util(data, name, vader_sentiment_scores, category, 
    plot_data, analysis_types, saves_dir):


    analysis_data = compute_vadersentiment(data, name, vader_sentiment_scores, saves_dir)    

    for analysis in analysis_types:
        if analysis == "review_level":   
            negation_pos_count_normalized = list(map(lambda x:x['negation_posemo'], analysis_data))        
        elif analysis == "sent_level":  
            negation_pos_count_normalized = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_sents"], analysis_data))       
        else:
            negation_pos_count_normalized = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_tokens"], analysis_data))        

        if analysis == "review_level":   
            negation_neg_count_normalized = list(map(lambda x:x['negation_negemo'], analysis_data))        
        elif analysis == "sent_level":  
            negation_neg_count_normalized = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_sents"], analysis_data))       
        else:
            negation_neg_count_normalized = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_tokens"], analysis_data))
          
           
        plot_data[analysis].append({
            "category": "positive - "+category+" review ",
            "review category": category,
            "text sentiment": "positive",
            "name": name,
            "value": np.mean(negation_pos_count_normalized),
            "sem_value": stats.sem(negation_pos_count_normalized),
            "all_samples_data": negation_pos_count_normalized
        })
        plot_data[analysis].append({
            "category": "negative - "+category+" review ",
            "review category": category,
            "text sentiment": "negative",
            "name": name,
            "value": np.mean(negation_neg_count_normalized),
            "sem_value": stats.sem(negation_neg_count_normalized),
            "all_samples_data": negation_neg_count_normalized
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
    myprint(f"args: {args}")   
    np.random.seed(args.seed_val)
    
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']

    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())

    saves_dir = os.path.join(args.saves_dir_name, "vader_pos_neg_negation_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "vader_pos_neg_negation_dist"

    vader_sentiment_scores = read_vader_sentiment_dict(args.vader_lexicon_path)
    datasets = json.loads(open(args.datasets_info_json, "r").read())

    for data in datasets:
        myprint(data)        
        compute_vadersentiment_util(data["positive"], data["name"], vader_sentiment_scores, "positive", 
            plot_data, analysis_types, saves_dir)

        compute_vadersentiment_util(data["negative"], data["name"], vader_sentiment_scores, "negative", 
            plot_data, analysis_types, saves_dir)

        print()
        print()

    pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))
