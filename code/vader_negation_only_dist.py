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
            # myprint(f"review: {rev}\nnegation count: {count}")
            rev_count += 1

        neg_words_count_arr = np.array(list(map(lambda x: x["negative_words_count"], neg_words_count)))
        neg_words_count_sent_normalized = np.array(list(map(lambda x: x["negative_words_count"]*1.0/x["sents_count"], neg_words_count)))
        neg_words_count_word_normalized = np.array(list(map(lambda x: x["negative_words_count"]*1.0/x["tokens_count"], neg_words_count)))

        # neg_words_sent_level = [1.0*x/y if y!=0 else 0 for x,y in zip(neg_words_count_arr, neg_words_count_sent_normalized)]
        # neg_words_word_level = [1.0*x/y if y!=0 else 0 for x,y in zip(neg_words_count_arr, neg_words_count_word_normalized)]

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
    seed_val = 23
    np.random.seed(seed_val)
    preload_flag = True

    saves_dir = os.path.join("saves", "negation")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    plot_save_prefix = "vader_negation_only_dist"
    
    seed_vals = [23]
    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())

    if not preload_flag:            
        dataset_files = json.loads(open("input.json").read())

        for data_file in dataset_files:
            for seed_val in seed_vals:
                myprint(f"Dataset: {data_file}")
                myprint(f"Seed val: {seed_val}")

                compute_negation_util(data_file["positive"], data_file["name"], seed_val, plot_data, 
                    "positive", analysis_types)
                compute_negation_util(data_file["negative"], data_file["name"], seed_val, plot_data, 
                    "negative", analysis_types)
                
                print()
                print()
        pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))        
    else:
        plot_data = pickle.load(open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "rb"))
    for analysis in analysis_types:
        amazon_data, non_amazon_data = util.filter_amazon(plot_data[analysis])
        # ylim_top = max(max([float(d["value"]) for d in amazon_data]), max([float(d["value"]) for d in non_amazon_data]))
        # ylim_top = 1.2*ylim_top

        ylim_top = max([float(d["value"]) for d in amazon_data])
        ylim_top = 1.7*ylim_top
        
        seaborn_plot_util.draw_grouped_barplot(amazon_data, "name", "value", 
            "category", os.path.join(saves_dir, plot_save_prefix+"_"+str(analysis)+"_amz"), 
            ylim_top=ylim_top,
            # y_axis_name="\#occurences",
            amazon_data_flag=True)

        ylim_top = max([float(d["value"]) for d in non_amazon_data])
        ylim_top = 1.7*ylim_top
        seaborn_plot_util.draw_grouped_barplot(non_amazon_data, "name", "value", 
            "category", os.path.join(saves_dir, plot_save_prefix+"_"+str(analysis)+"_non_amz"), ylim_top=ylim_top, 
            # y_axis_name="\#occurences"
            )
