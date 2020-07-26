import pickle
import spacy
import vader_negation_util
import numpy as np
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

def compute_negative_words_only(args: dict, seed_val: int): 

    np.random.seed(seed_val)

    dataset_file = args["data_filepath"]
    n_samples = None

    with open(dataset_file, "r") as fin:
        
        all_reviews = fin.readlines()

        if "n_samples" in args:
            n_samples = args["n_samples"]
        else:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=n_samples)       

        sampled_reviews = [all_reviews[idx] for idx in indices]

        neg_words_count = []
        rev_count = 0
        
        for rev in sampled_reviews:   
            rev=rev.strip("\n")
            neg_count = 0
            sents_count = 0
            
            doc = nlp(rev)
            tokens_count = len(doc)
            for sent in doc.sents:
                sents_count += 1
                words = sent.text.strip("\n").split()
                neg_count += vader_negation_util.negated(words)
            neg_words_count.append({
                "review": rev,
                "negative_words_count": neg_count,
                "sents_count": sents_count,
                "tokens_count": tokens_count
            })
            rev_count += 1

        neg_words_count_arr = np.array(list(map(lambda x: x["negative_words_count"], neg_words_count)))
        neg_words_count_sent_normalized = np.array(list(map(lambda x: x["negative_words_count"]*1.0/x["sents_count"], neg_words_count)))
        neg_words_count_word_normalized = np.array(list(map(lambda x: x["negative_words_count"]*1.0/x["tokens_count"], neg_words_count)))

        myprint(f"Avg number of negative words, review-level: {np.mean(neg_words_count_arr)}")
        myprint(f"Avg number of negative words, sent-level: {np.mean(neg_words_count_sent_normalized)}")
        myprint(f"Avg number of negative words, word-level: {np.mean(neg_words_count_word_normalized)}")

        return neg_words_count

def compute_negation_util(data_file, name, seed_val, plot_data, category, analysis_types):
    neg_words_count = compute_negative_words_only(data_file, seed_val)
    for analysis in analysis_types:
        if analysis == "word_level":   
            neg_words_normalized = list(map(lambda x: x["negative_words_count"]*1.0/x["tokens_count"], neg_words_count))
        elif analysis == "sent_level":   
            neg_words_normalized = list(map(lambda x: x["negative_words_count"]*1.0/x["sents_count"], neg_words_count))
        else:
            neg_words_normalized = list(map(lambda x: x["negative_words_count"], neg_words_count))
    
        plot_data[analysis].append({
            "name": name,
            "category": category,
            "value": np.mean(neg_words_normalized),
            "sem_value": stats.sem(neg_words_normalized),
            "all_samples_data": neg_words_normalized
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
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
   
    args = parser.parse_args()    
    myprint(f"args: {args}")   
    np.random.seed(args.seed_val)    

    saves_dir = os.path.join(open(args.saves_dir_name), "vader_negation")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    plot_save_prefix = "vader_negation_only_dist"
        
    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())

    dataset_files = json.loads(open(args.datasets_info_json).read())

    for data_file in dataset_files:            
        myprint(f"Dataset: {data_file}")            
        compute_negation_util(data_file["positive"], data_file["name"], args.seed_val, plot_data, 
            "positive", analysis_types)
        compute_negation_util(data_file["negative"], data_file["name"], args.seed_val, plot_data, 
            "negative", analysis_types)        
        print()
        print()
    pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))        