import os
import spacy
import numpy as np
import random
import torch

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def filter_amazon(plot_data): 
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    amazon_names = [val.lower() for val in amazon_names]   
    plot_data_amz = []
    plot_data_non_amz = []
    for d in plot_data:
        if d['name'].lower() in amazon_names:
            plot_data_amz.append(d)
        else:
            plot_data_non_amz.append(d)

    return plot_data_amz, plot_data_non_amz

# If there's a GPU available...
def get_device(device_no: int):
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:"+str(device_no))

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device

def get_samples(data, n_samples, seed_val=23):
    np.random.seed(seed_val)
    if n_samples == None:
        return data
        
    indices = np.random.choice(np.arange(len(data)), size=min(len(data),n_samples), replace=False)
    sampled_data = [data[idx] for idx in indices]
    return sampled_data

def filter_samples(data, max_token_length):
    selected_data = []
    for text in data:
        if len(text.split()) <= max_token_length:
            selected_data.append(text)
    return selected_data

def write_file(sents, out_file):
    with open(out_file, "w") as fout:
        for s in sents:
            fout.write(s.strip("\n")+"\n")


def read_file(filename):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            line = line.strip("\n")
            reviews.append(line)
        return reviews

def get_sents(reviews, min_no_of_tokens: int = 5):
    all_sents = []    
    for review in reviews:
        doc = nlp(review)
        for sent in doc.sents:
            tokens = tokenizer(sent.string.strip())
            if len(tokens) >= min_no_of_tokens:
                all_sents.append(sent.string.strip().strip("\n"))

    return all_sents