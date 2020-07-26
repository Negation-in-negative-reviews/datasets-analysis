import pickle
import os
import spacy
import numpy as np
from scipy import stats
import seaborn_plot_util
import pprint
import json
from pathlib import Path
import os
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")

def count_not_token(args: dict):
    all_reviews = []
    data_filepath = args["data_filepath"]
    review_analysis_data = []

    n_samples = None
    if "n_samples" in args:
        n_samples = args["n_samples"]

    with open(data_filepath, "r") as fin:
                
        for line in fin:
            all_reviews.append(line.strip("\n"))

        if n_samples == None:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=n_samples)
        selected_reviews = [all_reviews[idx] for idx in indices]

        for rev in selected_reviews:
            not_count = 0
            doc = nlp(rev)
            token_count = len(doc)
            sent_count = 0
            for sent in doc.sents:
                sent_count += 1
            not_count = rev.count(" not ")
            not_count += rev.count(" n't ")    
            not_count += rev.count("n't")
            # not_count /= n_tokens

            review_analysis_data.append({
                "review": rev,
                "not_count": not_count,
                "token_count": token_count,
                "sent_count": sent_count
            })

        not_count_sent_normalized = list(map(lambda x: x["not_count"]/x["sent_count"], review_analysis_data))
        not_count_token_normalized = list(map(lambda x: x["not_count"]/x["token_count"], review_analysis_data))        
        not_count_review_normalized = list(map(lambda x: x["not_count"], review_analysis_data))        

        myprint(f"not count review-level: {np.mean(not_count_review_normalized)}")
        myprint(f"not count sentence-level: {np.mean(not_count_sent_normalized)}")
        myprint(f"not count token-level: {np.mean(not_count_token_normalized)}")

        return not_count_sent_normalized, not_count_token_normalized, not_count_review_normalized

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
    parser.add_argument("--liwc_filepath",
                        default="/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic",
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    parser.add_argument("--preload_flag",
                        action='store_true',
                        help="Whether to use precomputed pickle files.")
    
    args = parser.parse_args()    
    np.random.seed(args.seed_val)

    saves_dir = os.path.join(args.saves_dir_name, "not_token_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)   

    save_prefix = "overall"
    plot_data_word_level = []
    plot_data_sent_level = []
    plot_data_review_level = []

    if not args.preload_flag:        
        datasets = json.loads(open(args.dataset_info_json, "r").read())
        for data in datasets:
            myprint(data)        
            not_count_sent_normalized, not_count_token_normalized, not_count_review_normalized = count_not_token(data["positive"])

            plot_data_word_level.append({
                "category": "positive reviews",
                "name": data["name"],
                "Average no of tokens": np.mean(not_count_token_normalized),
                "sem_value": stats.sem(not_count_token_normalized)
            })

            plot_data_sent_level.append({
                "category": "positive reviews",
                "name": data["name"],
                "Average no of tokens": np.mean(not_count_sent_normalized),
                "sem_value": stats.sem(not_count_sent_normalized)
            })

            plot_data_review_level.append({
                "category": "positive reviews",
                "name": data["name"],
                "Average no of tokens": np.mean(not_count_review_normalized),
                "sem_value": stats.sem(not_count_review_normalized)
            })

            not_count_sent_normalized, not_count_token_normalized, not_count_review_normalized = count_not_token(data["negative"])

            plot_data_word_level.append({
                "category": "negative reviews",
                "name": data["name"],
                "Average no of tokens": np.mean(not_count_token_normalized),
                "sem_value": stats.sem(not_count_token_normalized)
            })

            plot_data_sent_level.append({
                "category": "negative reviews",
                "name": data["name"],
                "Average no of tokens": np.mean(not_count_sent_normalized),
                "sem_value": stats.sem(not_count_sent_normalized)
            })

            plot_data_review_level.append({
                "category": "negative reviews",
                "name": data["name"],
                "Average no of tokens": np.mean(not_count_review_normalized),
                "sem_value": stats.sem(not_count_review_normalized)
            })
            
            print()
            print()

        pickle.dump({
            "plot_data_word_level": plot_data_word_level,
            "plot_data_sent_level": plot_data_sent_level,
            "plot_data_review_level": plot_data_review_level
        }, open(os.path.join(saves_dir, save_prefix+"_dist.pickle"), "wb"))
    else:
        temp_json = pickle.load(open(os.path.join(saves_dir, save_prefix+"_dist.pickle"), "rb"))
        plot_data_word_level = temp_json["plot_data_word_level"]
        plot_data_sent_level = temp_json["plot_data_sent_level"]
        plot_data_review_level = temp_json["plot_data_review_level"]

    seaborn_plot_util.draw_grouped_barplot(plot_data_word_level, "name", "Average no of tokens", 
    "category", os.path.join(saves_dir, save_prefix+"_word_level.png"))
    seaborn_plot_util.draw_grouped_barplot(plot_data_sent_level, "name", "Average no of tokens", 
    "category", os.path.join(saves_dir, save_prefix+"_sent_level.png"))
    seaborn_plot_util.draw_grouped_barplot(plot_data_review_level, "name", "Average no of tokens", 
    "category", os.path.join(saves_dir, save_prefix+"_review_level.png"))
