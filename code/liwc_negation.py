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
pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")

def compute_liwc(data, result, class_id, cluster_result, 
categories, category_reverse):
    
    data_filepath = data["data_filepath"]
    save_pickle_path = data["save_pickle_path"]
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

        indices = np.random.choice(np.arange(len(all_reviews)), size=n_samples)
        selected_reviews = [all_reviews[idx] for idx in indices]
        count = 0
        for rev in selected_reviews:
            # if count%1000 == 0:
            #     print(count)
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

        # myprint(f"Negemo words: {list(map(lambda x: x['negemo_words'], review_level_liwc))}")
        # myprint(f"Posemo words: {list(map(lambda x: x['posemo_words'], review_level_liwc))}")

        pickle.dump({
            "negation_analysis_data": review_level_liwc
        }, open(save_pickle_path, "wb"))
        


if __name__ == "__main__": 
    seed_val = 23
    np.random.seed(seed_val)

    liwc_filepath = "/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic"
    result, class_id, cluster_result, categories, category_reverse = liwc_util.load_liwc(liwc_filepath)
    
    datasets = [        
        {
            "data_filepath": "/data/madhu/stanford-sentiment-treebank/matched_data/neg_reviews.txt",
            "save_pickle_path": "pickle_saves/sst_neg_liwc_negation.pickle",
            # "n_samples": 100
        },
        {
            "data_filepath": "/data/madhu/stanford-sentiment-treebank/matched_data/pos_reviews.txt",
            "save_pickle_path": "pickle_saves/sst_pos_liwc_negation.pickle",
            # "n_samples": 100
        },
        {
            "data_filepath": "/data/madhu/yelp/yelp_processed_data/review.0",            
            "save_pickle_path": "pickle_saves/yelp_neg_liwc_negation_dist.pickle",
            "n_samples": 5000
        },
        {
            "data_filepath": "/data/madhu/yelp/yelp_processed_data/review.1",            
            "save_pickle_path": "pickle_saves/yelp_pos_liwc_negation_dist.pickle",
            "n_samples": 5000
        },        
        # {
        #     "data_filepath": "/data/madhu/imdb_dataset/processed_data/pos_reviews_train",            
        #     "save_pickle_path": "pickle_saves/imdb_pos_5k_liwc_negation_dist.pickle",
        #     "n_samples": 5000
        # },
        # {
        #     "data_filepath": "/data/madhu/imdb_dataset/processed_data/neg_reviews_train",            
        #     "save_pickle_path": "pickle_saves/imdb_neg_5k_liwc_negation_dist.pickle",
        #     "n_samples": 5000
        # }         
    ]

    for data in datasets:
        myprint(data)        
        compute_liwc(data, result, class_id, 
            cluster_result, categories, category_reverse)
        print()
        print()