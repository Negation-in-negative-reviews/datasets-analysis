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

if __name__ == "__main__": 
    seed_val = 23
    np.random.seed(seed_val)

    saves_dir = os.path.join("saves", "dataset_length_dist_full")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "dataset_length_dist"

    dataset_names = [
        "Yelp", "SST", "IMDB", "Tripadvisor", "Cellphones", 
        "Automotive", "Luxury beauty", "Pet supplies", "Sports"
    ]
    data = {
        "positive": {
            "tokens": {
                "Yelp": 110.8154537, 
                "SST": 21.55386983, 
                "IMDB": 275.512, 
                "Tripadvisor": 191.5470864, 
                "Cellphones": 56.01688558, 
                "Automotive": 36.34252001, 
                "Luxury beauty": 101.0168386, 
                "Pet supplies": 53.69575864, 
                "Sports": 51.45845473
            },
            "sentences": {
                "Yelp": 8.777560505, 
                "SST": 1.139841689, 
                "IMDB": 14.22752, 
                "Tripadvisor": 12.30545737, 
                "Cellphones": 4.307873998, 
                "Automotive": 3.026564392, 
                "Luxury beauty": 6.81866155, 
                "Pet supplies": 4.048211006, 
                "Sports": 3.98356711
            }
        },
        "negative": {
            "tokens": {
                "Yelp": 149.1341537, 
                "SST": 22.11186114, 
                "IMDB": 268.4688,
                "Tripadvisor": 243.8720003,
                "Cellphones": 61.87677279, 
                "Automotive": 58.75849589,
                "Luxury beauty": 102.3535769,
                "Pet supplies": 63.48955889, 
                "Sports": 67.84724381
            },
            "sentences": {
                "Yelp": 11.11709038, 
                "SST": 1.195756991, 
                "IMDB": 15.07896, 
                "Tripadvisor": 15.23182561, 
                "Cellphones": 4.688768593, 
                "Automotive": 4.457281113, 
                "Luxury beauty": 6.992794647, 
                "Pet supplies": 4.676691155, 
                "Sports": 5.003716284
            }
        }
    }

    plot_data = []
    for name in dataset_names:
        for cat in ["positive", "negative"]:
            plot_data.append({
                "name": name,
                "#tokens": data[cat]["tokens"][name],
                "#sentences": data[cat]["sentences"][name],
                "sem_value": -1,
                "category": cat
            })

    for key in ["#tokens", "#sentences"]:
        plot_data_amz, plot_data_non_amz = util.filter_amazon(plot_data)
        ylim_top = max([float(d[key]) for d in plot_data_amz])
        ylim_top = 1.2*ylim_top
        seaborn_plot_util.draw_grouped_barplot(plot_data_amz, "name", key, "category",
            os.path.join(saves_dir, plot_save_prefix+"_"+key+"_amz.pdf"), y_axis_name=key, ylim_top=ylim_top)
        
        ylim_top = max([float(d[key]) for d in plot_data_non_amz])
        ylim_top = 1.2*ylim_top
        seaborn_plot_util.draw_grouped_barplot(plot_data_non_amz, "name", key, "category",
            os.path.join(saves_dir, plot_save_prefix+"_"+key+"_non_amz.pdf"), y_axis_name=key, ylim_top=ylim_top)
