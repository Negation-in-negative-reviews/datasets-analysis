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

def compute_dataset_length(data):

    data_filepath = data["data_filepath"]
    all_reviews = []
    
    with open(data_filepath, "r") as fin:
        for line in fin:
            all_reviews.append(line.strip("\n"))

        n_samples = None

        if "n_samples" in data:
            n_samples = data["n_samples"]
        else:
            n_samples = len(all_reviews)

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
        token_count_review_level = list(map(lambda x:x['token_count'], analysis_data))
        token_count_sent_level = list(map(lambda x:1.0*x['token_count']/x["sent_count"], analysis_data))        

        return token_count_review_level, token_count_sent_level            

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
                        required=True,
                        help="")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    parser.add_argument("--preload_flag",
                        action='store_true',
                        help="Whether to run training.")
    
    args = parser.parse_args()

    np.random.seed(args.seed_val)
    
    saves_dir = os.path.join(args.saves_dir_name, "dataset_length")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)           
    plot_data={
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())
    plot_save_prefix = "dataset_length_dist"

    if not args.preload_flag:
        datasets = json.loads(open(args.datasets_info_json, "r").read())
        for data in datasets:
            myprint(data)        
            token_count = {}
            token_count["review_level"], token_count["sent_level"] = compute_dataset_length(data["positive"])       

            for a_type in analysis_types:
                plot_data[a_type].append({
                    "category": "positive reviews",
                    "name": data["name"],
                    "value": np.mean(token_count[a_type]),
                    "sem_value": stats.sem(token_count[a_type]),
                    "all_samples_data": token_count[a_type]
                })
            token_count = {}
            token_count["review_level"], token_count["sent_level"] = compute_dataset_length(data["negative"])

            for a_type in analysis_types:
                plot_data[a_type].append({
                    "category": "negative reviews",
                    "name": data["name"],
                    "value": np.mean(token_count[a_type]),
                    "sem_value": stats.sem(token_count[a_type]),
                    "all_samples_data": token_count[a_type]
                })

            print()
            print()
            
        pickle.dump({
            "plot_data": plot_data
        }, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))
    else:
        plot_data = pickle.load(open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "rb"))["plot_data"]

    # for a_type in analysis_types:
    #     plot_data_amz, plot_data_non_amz = util.filter_amazon(plot_data[a_type])

    #     ylim_top = max([float(d["value"]) for d in plot_data_non_amz])
    #     ylim_top = 1.2*ylim_top

    #     seaborn_plot_util.draw_grouped_barplot(plot_data_non_amz, "name", "value",
    #                                 "category", os.path.join(saves_dir, plot_save_prefix+"_"+a_type+"_non_amz"),
    #                                 figsize=(15, 6), position=(0.08, 0.08, 0.6, 0.9),
    #                                 ylim_top=ylim_top, bbox_to_anchor=(1, 0.5, 0.2, 0.5))
        
    #     ylim_top = max([float(d["value"]) for d in plot_data_amz])
    #     ylim_top = 1.7*ylim_top
        
    #     seaborn_plot_util.draw_grouped_barplot(plot_data_amz, "name", "value", 
    #     "category", os.path.join(saves_dir, plot_save_prefix+"_"+a_type+"_amz"),ylim_top=ylim_top)

        