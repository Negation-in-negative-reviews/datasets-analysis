import pickle
import argparse
import pandas as pd
from scipy import stats
import numpy as np
import pprint 

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def read_pickle_file(filename):

    analysis_types = [
        # "word_level", 
        "sent_level", 
        "review_level"
    ]
    data = pickle.load(open(filename, "rb"))
    # data = data["plot_data"]
    stats_data = pd.DataFrame(columns=["name", "analysis_type", "statistic", "pvalue"])
    # sentiment_categories = ["anger", "sad", "posemo", "negemo"]
    # sentiment_key = "liwc_category"
    sentiment_categories = ["positive", "negative"]
    sentiment_key = "text sentiment"

    review_key = "review category"
    # review_key = "category"
    for a_type in analysis_types:
        df = pd.DataFrame(data[a_type])   
        # print(df.head(5))     
        dataset_names = df["name"].unique()
        for name in dataset_names:
            for sent_cat in sentiment_categories:                
                df_name = df[df["name"]==name]
                df_sent = df_name[df_name[sentiment_key]==sent_cat]
                df_sent = df_name
                pos_df_counts = df_sent[df_sent[review_key] == "positive"]["all_samples_data"].to_list()
                neg_df_counts = df_sent[df_sent[review_key] == "negative"]["all_samples_data"].to_list()                
                min_len = min(len(pos_df_counts[0]), len(neg_df_counts[0]))
                statistic, pvalue = stats.ttest_ind(pos_df_counts[0][:min_len],neg_df_counts[0][:min_len])
                stats_data = stats_data.append({
                    "name": name,
                    "analysis_type": a_type,
                    "statistic": statistic,
                    "pvalue": pvalue,
                    # "liwc_category": sent_cat
                }, ignore_index=True)

    return stats_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--pickle_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    args = parser.parse_args()

    stats_data = read_pickle_file(args.pickle_file)
    print(stats_data)