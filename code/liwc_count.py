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
import json
import os
from pathlib import Path
import seaborn_plot_util
import pprint
import util
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")

def compute_liwc(plot_data, plot_data_negation, data, dataset_name, review_category, 
    required_categories, result, class_id, cluster_result, 
    categories, category_reverse, analysis_types):
    
    data_filepath = data["data_filepath"]

    n_samples = None
    if "n_samples" in data:
        n_samples = data["n_samples"]
    
    reviews = util.read_file(data_filepath)
    selected_reviews = util.get_samples(reviews, n_samples)
    all_reviews_data = []
    for rev in selected_reviews:
        doc = nlp(rev)            
        token_count = len(doc)
        review_data = {}
        sent_count = 0
        for sent in doc.sents:  
            sent_count += 1
        negation_pos = 0
        negation_neg = 0
        for cat in required_categories:
            review_data[cat] = 0
            
        for idx,token in enumerate(doc):
            for cat in required_categories:
                for pattern in cluster_result[category_reverse[cat]]:
                    if (pattern.endswith("*") and token.text.startswith(pattern[:-1])) or (pattern==token.text):
                        review_data[cat] = review_data.get(cat, 0) + 1      
                        if cat == "posemo" and idx>0:
                            if doc[idx-1].text in vader_negation_util.NEGATE:
                                negation_pos += 1
                        elif cat == "negemo" and idx>0:
                            if doc[idx-1].text in vader_negation_util.NEGATE:
                                negation_neg += 1
            
        review_data["total_no_of_tokens"] = token_count
        review_data["total_no_of_sents"] = sent_count
        review_data["negation_posemo"] = negation_pos
        review_data["negation_negemo"] = negation_neg
        
        all_reviews_data.append(review_data)

    category_counts = {}
    for cat in required_categories:       
        category_counts["word_level"] = list(map(lambda x:1.0*x[cat]/x["total_no_of_tokens"], all_reviews_data))
        category_counts["sent_level"] = list(map(lambda x:1.0*x[cat]/x["total_no_of_sents"], all_reviews_data))
        category_counts["review_level"] = list(map(lambda x:x[cat], all_reviews_data))
        
        for a_type in analysis_types:                
            plot_data[a_type].append({
                # "category": "negative - "+review_category+" review ",
                "review category": review_category,
                "liwc_category": cat,
                "name": dataset_name,
                "value": np.mean(category_counts[a_type]),
                "sem_value": stats.sem(category_counts[a_type]),
                "all_samples_data": category_counts[a_type]
            })               
    
    category_counts = {}
    for cat in ["negation_posemo", "negation_negemo"]:       
        category_counts["word_level"] = list(map(lambda x:1.0*x[cat]/x["total_no_of_tokens"], all_reviews_data))
        category_counts["sent_level"] = list(map(lambda x:1.0*x[cat]/x["total_no_of_sents"], all_reviews_data))
        category_counts["review_level"] = list(map(lambda x:x[cat], all_reviews_data))
        
        for a_type in analysis_types:                
            plot_data_negation[a_type].append({
                # "category": "negative - "+review_category+" review ",
                "review category": review_category,
                "negation_category": cat,
                "name": dataset_name,
                "value": np.mean(category_counts[a_type]),
                "sem_value": stats.sem(category_counts[a_type]),
                "all_samples_data": category_counts[a_type]
            })               


    return plot_data, plot_data_negation

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
    parser.add_argument("--liwc_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    # parser.add_argument("--preload_flag",
    #                     action='store_true',
    #                     help="Whether to run training.")
    
    args = parser.parse_args()  
    myprint(f"args: {args}")     
    np.random.seed(args.seed_val)

    saves_dir = os.path.join(args.saves_dir_name, "liwc_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "liwc"

    analysis_types = [
        "sent_level", 
        "review_level", 
        "word_level"
    ]
    required_categories = [
        "posemo", 
        "negemo", 
        "anger", 
        "sad", 
    ]

    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']

    # if not args.preloadflag:
    result, class_id, cluster_result, categories, category_reverse = liwc_util.load_liwc(args.liwc_filepath)            
    datasets = json.loads(open(args.datasets_info_json, "r").read())
    plot_data = {}
    plot_data_negation = {}
    for a_type in analysis_types:
        plot_data[a_type] = []
        plot_data_negation[a_type] = []
    for data in datasets:
        myprint(data)
        plot_data, plot_data_negation = compute_liwc(plot_data, plot_data_negation, data["positive"], data["name"], "positive", 
            required_categories, result, class_id, cluster_result, categories, category_reverse, analysis_types)
        plot_data, plot_data_negation = compute_liwc(plot_data, plot_data_negation, data["negative"], data["name"], "negative", 
            required_categories, result, class_id, cluster_result, categories, category_reverse, analysis_types)
        
    pickle_save_dir = os.path.join(saves_dir, "all")
    Path(pickle_save_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(plot_data, open(os.path.join(pickle_save_dir, "liwc_dist_data.pickle"), "wb"))
    pickle.dump(plot_data_negation, open(os.path.join(pickle_save_dir, "liwc_dist_negation_data.pickle"), "wb"))
    # else:
    #     pickle_save_dir = os.path.join(saves_dir, "all")
    #     Path(pickle_save_dir).mkdir(parents=True, exist_ok=True)
    #     plot_data = pickle.load(open(os.path.join(pickle_save_dir, "liwc_dist_data.pickle"), "rb"))  
    
    # plot_categories = [
    #     ["posemo","negemo"], 
    #     ["anger", "sad"]
    # ]
    # for analysis in analysis_types:
    #     plot_data_df = pd.DataFrame(plot_data[analysis])
    #     plot_data_pos_neg_amz = []
    #     plot_data_pos_neg_non_amz = []
    #     colors =[
    #         [(114/255, 200/255, 117/255),(209/255, 68/255, 68/255)]*2,
    #         [(67/255, 144/255, 188/255),(141/255, 190/255, 216/255)]*2,
    #     ]  
    #     for idx,plot_cat in enumerate(plot_categories):
    #         plot_data_cat = plot_data_df[plot_data_df["liwc_category"].isin(plot_cat)]
    #         plot_data_cat = plot_data_cat.to_dict('records')            
    #         plot_data_cat_amz, plot_data_cat_non_amz = util.filter_amazon(plot_data_cat)
        
    #         ylim_top = max([float(d["value"]) for d in plot_data_cat_amz])
    #         ylim_top = 1.7*ylim_top
    #         seaborn_plot_util.draw_grouped_barplot_four_subbars_liwc(plot_data_cat_amz, colors[idx],
    #             "name", "value", "review category", 
    #             os.path.join(saves_dir, plot_save_prefix+"_"+"_".join(plot_cat)+"_"+str(analysis)+"_amz"),
    #             ylim_top=ylim_top, liwc_cats=plot_cat, amazon_data_flag=True)

    #         ylim_top = max([float(d["value"]) for d in plot_data_cat_non_amz])
    #         ylim_top = 1.7*ylim_top
    #         seaborn_plot_util.draw_grouped_barplot_four_subbars_liwc(plot_data_cat_non_amz, colors[idx], 
    #             "name", "value", "review category", 
    #             os.path.join(saves_dir, plot_save_prefix+"_"+"_".join(plot_cat)+"_"+str(analysis)+"_non_amz"),
    #             ylim_top=ylim_top, liwc_cats=plot_cat)
