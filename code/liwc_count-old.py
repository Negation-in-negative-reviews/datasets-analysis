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
pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")

def compute_liwc(data, dataset_name, category, required_categories, result, class_id, cluster_result, 
    categories, category_reverse, analysis_type):
    
    data_filepath = data["data_filepath"]
    saves_dir = os.path.join("saves", dataset_name)
    Path(saves_dir).mkdir(parents=True, exist_ok=True)    

    # save_image_path_pdf = os.path.join(saves_dir, category+"_liwc_dist_"+analysis_type+".pdf")
    # save_image_path_png = os.path.join(saves_dir, category+"_liwc_dist_"+analysis_type+".pdf")

    save_pickle_path = os.path.join(saves_dir, category+"_liwc_dist_"+analysis_type+".pickle")

    n_samples = None
    if "n_samples" in data:
        n_samples = data["n_samples"]
    
    all_reviews_data = []    
    with open(data_filepath, "r") as f:
        all_reviews = []
        for rev in f.readlines():
            rev = rev.strip("\n")
            all_reviews.append(rev)

        if n_samples == None:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=n_samples)
        selected_reviews = [all_reviews[idx] for idx in indices]

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

        cluster_counts = {}

        for cat in required_categories:       
            category_counts_word_level = list(map(lambda x:1.0*x[cat]/x["total_no_of_tokens"], all_reviews_data))
            category_counts_sent_level = list(map(lambda x:1.0*x[cat]/x["total_no_of_sents"], all_reviews_data))
            category_counts_review_level = list(map(lambda x:x[cat], all_reviews_data))

            if analysis_type == "word-level": 
                cluster_counts[cat] = {
                    "average no of tokens": np.mean(category_counts_word_level),
                    "sem_value": stats.sem(category_counts_word_level)
                }
            elif analysis_type == "sent-level":
                cluster_counts[cat] = {                    
                    "average no of tokens": np.mean(category_counts_sent_level),
                    "sem_value": stats.sem(category_counts_sent_level)
                }
            elif analysis_type == "review-level":
                cluster_counts[cat] = {                    
                    "average no of tokens": np.mean(category_counts_review_level),
                    "sem_value": stats.sem(category_counts_review_level)
                }

        tokens_per_sent = list(map(lambda x:x['total_no_of_tokens']/x["total_no_of_sents"], all_reviews_data)) 
        
        avg_tokens_per_sent = np.mean(tokens_per_sent)        
        avg_tokens_per_sent_sem = stats.sem(tokens_per_sent)

        avg_tokens_per_review = np.mean(list(map(lambda x:x['total_no_of_tokens'], all_reviews_data)))
        avg_tokens_per_review_sem = stats.sem(list(map(lambda x:x['total_no_of_tokens'], all_reviews_data)))

        # print("save image path inside draw_histogram_liwc()", save_image_path_pdf)
        # print("save pickle path inside draw_histogram_liwc()", save_pickle_path)


        result = {
            "all_reviews_data": all_reviews_data,
            "cluster_counts": cluster_counts,
            "name": dataset_name,
            "analysis_type": analysis_type,
            "avg_tokens_per_sent": avg_tokens_per_sent,
            "avg_tokens_per_sent_sem": avg_tokens_per_sent_sem,
            "avg_tokens_per_review": avg_tokens_per_review,
            "avg_tokens_per_review_sem": avg_tokens_per_review_sem
        }
        # pickle.dump(result, open(save_pickle_path, "wb"))

        print("avg_tokens_per_sentence: ", avg_tokens_per_sent)
        print("avg_tokens_per_review: ", avg_tokens_per_review)

        # seaborn_plot_util.draw_histogram(analysis_type, "category", "average no of tokens", save_image_path_png)
        # seaborn_plot_util.draw_histogram(analysis_type, "category", "average no of tokens", save_image_path_pdf)

        return result

        # draw_histogram_liwc(avg_cluster_counts, cluster_sem_vals, required_categories, save_image_path_pdf)
        # draw_histogram_liwc(avg_cluster_counts, cluster_sem_vals, required_categories, save_image_path_png)

def compute_liwc_util(plot_data, name, review_data, liwc_category, review_category):
    plot_data.append({
        "liwc_category": liwc_category,
        "name": name,
        "review category": review_category,
        "value": review_data[liwc_category]["average no of tokens"],
        "sem_value": review_data[liwc_category]["sem_value"]
    })

if __name__ == "__main__": 
    seed_val = 23
    np.random.seed(seed_val)

    liwc_filepath = "/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic"

    '''
        result: a dictionary that maps each word to the LIWC cluster ids
            that it belongs to
        class_id: a dict that maps LIWC cluster to category id,
            this does not seem useful, here for legacy reasons
        cluster_result: a dict that maps LIWC cluster id to all words
            in that cluster
        categories: a dict that maps LIWC cluster to its name
        category_reverse: a dict that maps LIWC cluster name to its id
    '''

    preloadflag = True
    saves_dir = os.path.join("saves", "liwc")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "liwc_dist"
    plot_data = {}
    analysis_types = [
        "sent-level", 
        "review-level", 
        "word-level"
    ]
    required_categories = [
        "posemo", 
        "negemo", 
        "anger", 
        "sad", 
    ]

    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']

    if not preloadflag:
        result, class_id, cluster_result, categories, category_reverse = liwc_util.load_liwc(liwc_filepath)            
        datasets = json.loads(open("input.json", "r").read())
        for analysis in analysis_types:
            # plot_data[analysis] = {}
            for cat in required_categories:
                # plot_data[analysis][cat] = []
                plot_data = []
                for data in datasets:
                    myprint(data)
                    myprint(f"analysis: {analysis}")
                    pos_result = compute_liwc(data["positive"], data["name"], "positive", required_categories, result, class_id, 
                        cluster_result, categories, category_reverse, analysis)
                    neg_result = compute_liwc(data["negative"], data["name"], "negative", required_categories, result, class_id, 
                        cluster_result, categories, category_reverse, analysis)
                    compute_liwc_util(plot_data, data["name"], pos_result["cluster_counts"], cat, "positive")
                    compute_liwc_util(plot_data, data["name"], neg_result["cluster_counts"], cat, "negative")
                
                seaborn_plot_util.draw_grouped_barplot(plot_data, "name", "value", "review category", 
                    os.path.join(saves_dir, plot_save_prefix+"_"+str(cat)+"_"+str(analysis)+".pdf"))
                pickle_save_dir = os.path.join(saves_dir, analysis, cat)
                Path(pickle_save_dir).mkdir(parents=True, exist_ok=True)
                pickle.dump(plot_data, open(os.path.join(pickle_save_dir, "liwc_dist_data.pickle"), "wb"))
    else:
        for analysis in analysis_types:
            plot_data_pos_neg_amz = []
            plot_data_pos_neg_non_amz = []
            for cat in required_categories:
                pickle_save_dir = os.path.join(saves_dir, analysis, cat)
                plot_data = pickle.load(open(os.path.join(pickle_save_dir, "liwc_dist_data.pickle"), "rb"))                
                plot_data_amz, plot_data_non_amz = util.filter_amazon(plot_data)
                plot_data_pos_neg_amz += plot_data_amz
                plot_data_pos_neg_non_amz += plot_data_non_amz
            
            ylim_top = max([float(d["value"]) for d in plot_data_pos_neg_amz])
            ylim_top = 1.4*ylim_top
            seaborn_plot_util.draw_grouped_barplot_four_subbars_liwc(plot_data_pos_neg_amz, "name", "value", "review category", 
                os.path.join(saves_dir, plot_save_prefix+"_"+"_".join(required_categories)+"_"+str(analysis)+"_amz.pdf"),
                ylim_top=ylim_top)

            ylim_top = max([float(d["value"]) for d in plot_data_pos_neg_non_amz])
            ylim_top = 1.4*ylim_top
            seaborn_plot_util.draw_grouped_barplot_four_subbars_liwc(plot_data_pos_neg_non_amz, "name", "value", "review category", 
                os.path.join(saves_dir, plot_save_prefix+"_"+"_".join(required_categories)+"_"+str(analysis)+"_non_amz.pdf"),
                ylim_top=ylim_top)