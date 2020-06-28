import spacy
import argparse
import numpy as np
import util
import pickle
import os
from pathlib import Path
import vader_negation_util
import json
import pprint
from scipy import stats
import seaborn_plot_util_old

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load('en_core_web_md')
VADER_LEXICON_PATH = "/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"

def count_negation(text, vader_sentiment_scores):
    doc = nlp(text)
    negation_count = 0
    dep_dict = {}
    sent_count = 0
    pos_negation_count = 0
    neg_negation_count = 0
    for sent in doc.sents:
        sent_count += 1
        doc_sent = nlp(sent.text)
        for token in doc_sent: 
            dep_dict[token.text] = [(child.text, child.dep_) for child in token.children]  
            if token.dep_ == "neg":
                negation_count += 1
                if token.head.pos_ == "AUX":
                    if token.head.dep_ in ["acomp", "advmod"] and token.head.head.text in vader_sentiment_scores:
                        sent_score = vader_sentiment_scores[token.head.head.text]
                        if sent_score >= 1:
                            pos_negation_count += 1
                        elif sent_score <= -1:
                            neg_negation_count += 1
                elif token.head.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and token.head.text in vader_sentiment_scores:
                    sent_score = vader_sentiment_scores[token.head.text]
                    if sent_score >= 1:
                        pos_negation_count += 1
                    elif sent_score <= -1:
                        neg_negation_count += 1
    
    negation_count_dict = {
        "word_level": 1.0*negation_count/len(doc),
        "sent_level": 1.0*negation_count/sent_count,
        "review_level": negation_count
    }
    pos_negation_count_dict = {
        "word_level": 1.0*pos_negation_count/len(doc),
        "sent_level": 1.0*pos_negation_count/sent_count,
        "review_level": pos_negation_count
    }
    neg_negation_count_dict = {
        "word_level": 1.0*neg_negation_count/len(doc),
        "sent_level": 1.0*neg_negation_count/sent_count,
        "review_level": neg_negation_count
    }
    return negation_count_dict, pos_negation_count_dict, neg_negation_count_dict, dep_dict


if __name__ == "__main__":
    saves_dir = os.path.join("saves", "negation_dep_parsing_2_hops")
    Path(saves_dir).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(saves_dir, "all")).mkdir(parents=True, exist_ok=True) 
    datasets = json.loads(open("input.json").read())
    preload_flag = True
    plot_save_prefix = "pos_neg_negation_depparsing"
    analysis_types = [
        "word_level", 
        "sent_level", 
        "review_level"
    ]
    
    if not preload_flag:
        seed_val = 23
        np.random.seed(seed_val)

        vader_sentiment_scores = vader_negation_util.read_vader_sentiment_dict(VADER_LEXICON_PATH)

        negation_count_data = {}
        pos_negation_count_data = {}
        neg_negation_count_data = {}

        selected_samples = {}
        for data in datasets:
            myprint(data)
            selected_samples[data["name"]] = {}
            for category in ["positive", "negative"]:
                texts = util.read_file(data[category]["data_filepath"])
                n_samples = None
                if "n_samples" in data[category]:
                    n_samples = data[category]["n_samples"]
                selected_texts = util.get_samples(texts, n_samples)
                selected_samples[data["name"]][category] = selected_texts

        plot_data = {}
        plot_data_overall_negation = {}
        
        for a_type in analysis_types:
            plot_data[a_type] = []
            plot_data_overall_negation[a_type] = []

        for data in datasets:
            dep_data = {}
            for category in ["positive", "negative"]:
                dep_data[category] = []
                negation_count_data = {}
                pos_negation_count_data = {}
                neg_negation_count_data = {}
                for a_type in analysis_types:
                    negation_count_data[a_type] = []
                    pos_negation_count_data[a_type] = []
                    neg_negation_count_data[a_type] = []

                selected_texts = selected_samples[data["name"]][category]
                for text in selected_texts:
                    negation_count_dict, pos_negation_count_dict, neg_negation_count_dict, dep_dict = count_negation(text, 
                        vader_sentiment_scores)
                    for a_type in analysis_types:
                        negation_count_data[a_type].append(negation_count_dict[a_type])
                        pos_negation_count_data[a_type].append(pos_negation_count_dict[a_type])
                        neg_negation_count_data[a_type].append(neg_negation_count_dict[a_type])                    

                dep_data[category].append({
                    "text": text,
                    "dep_info": dep_dict                        
                })
                for a_type in analysis_types:
                    plot_data[a_type].append({
                        "category": "negative - "+category+" review ",
                        "review category": category,
                        "text sentiment": "negative",
                        "name": data["name"],
                        "value": np.mean(neg_negation_count_data[a_type]),
                        "sem_value": stats.sem(neg_negation_count_data[a_type]),
                        "all_samples_data": neg_negation_count_data[a_type]
                    })
                    plot_data[a_type].append({
                        "category": "positive - "+category+" review ",
                        "review category": category,
                        "text sentiment": "positive",
                        "name": data["name"],
                        "value": np.mean(pos_negation_count_data[a_type]),
                        "sem_value": stats.sem(pos_negation_count_data[a_type]),
                        "all_samples_data": pos_negation_count_data[a_type]
                    })
                    plot_data_overall_negation[a_type].append({
                        "category": category,
                        "name": data["name"],
                        "value": np.mean(negation_count_data[a_type]),
                        "sem_value": stats.sem(negation_count_data[a_type]),
                        "all_samples_data": negation_count_data[a_type]
                    })
            pickle.dump(dep_data, 
                open(os.path.join(saves_dir, data["name"]+"_depparsing_data.pickle"), "wb")
            )
                    
        pickle.dump(plot_data, 
            open(os.path.join(saves_dir, "all", plot_save_prefix+".pickle"), "wb")
        )
        pickle.dump(plot_data_overall_negation, 
            open(os.path.join(saves_dir, "all", "overall.pickle"), "wb")
        )
    else:
        plot_data = pickle.load(open(os.path.join(saves_dir, "all", plot_save_prefix+".pickle"), "rb"))
        plot_data_overall_negation = pickle.load(open(os.path.join(saves_dir, "all", "overall.pickle"), "rb"))
    
    for analysis in analysis_types:   
        myprint(analysis)     
        amazon_data, non_amazon_data = util.filter_amazon(plot_data[analysis])
        ylim_top = max([float(d["value"]) for d in non_amazon_data])
        ylim_top = 1.7*ylim_top

        seaborn_plot_util_old.draw_grouped_barplot_four_subbars(non_amazon_data, "name", "value", 
            "category", 
            os.path.join(saves_dir, "all", 
            plot_save_prefix+"_"+str(analysis)+"_non_amz"),
            ylim_top=ylim_top,
            negation=True,
            y_axis_name="#occurences")
        
        ylim_top =  max([float(d["value"]) for d in amazon_data])
        ylim_top = 1.7*ylim_top
        seaborn_plot_util_old.draw_grouped_barplot_four_subbars(amazon_data, "name", "value", 
            "category", 
            os.path.join(saves_dir, "all", 
            plot_save_prefix+"_"+str(analysis)+"_amz"),
            ylim_top=ylim_top,
            negation=True,
            y_axis_name="#occurences", amazon_data_flag=True)

    print("Overall negation\n\n")
    for analysis in analysis_types:   
        print()
        myprint(analysis)     
        print("-----------")
        amazon_data, non_amazon_data = util.filter_amazon(plot_data_overall_negation[analysis])
        ylim_top = max([float(d["value"]) for d in non_amazon_data])
        ylim_top = 1.7*ylim_top

        seaborn_plot_util_old.draw_grouped_barplot(non_amazon_data, "name", "value", 
            "category", 
            os.path.join(saves_dir, "all", "overall_negation_"+str(analysis)+"_non_amz"),
            ylim_top=ylim_top,
            y_axis_name="#occurences")
        
        ylim_top =  max([float(d["value"]) for d in amazon_data])
        ylim_top = 1.7*ylim_top
        seaborn_plot_util_old.draw_grouped_barplot(amazon_data, "name", "value", 
            "category", 
            os.path.join(saves_dir, "all", "overall_negation_"+str(analysis)+"_amz"),
            ylim_top=ylim_top,
            y_axis_name="#occurences", amazon_data_flag=True)
