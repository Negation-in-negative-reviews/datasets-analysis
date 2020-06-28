import spacy
import argparse
import numpy as np
import util
import pickle
import os
from pathlib import Path

nlp = spacy.load('en_core_web_md')

def count_negation(text):
    doc = nlp(text)
    neg_count = 0
    dep_dict = {}
    sent_count = 0
    for sent in doc.sents:
        sent_count += 1
        doc_sent = nlp(sent.text)
        for token in doc_sent:        
            dep_dict[token.text] = [(child.text, child.dep_) for child in token.children]
            if token.dep_ == "neg":
                neg_count += 1
    neg_count_dict = {
        "word_level": 1.0*neg_count/len(doc),
        "sent_level": 1.0*neg_count/sent_count,
        "review_level": neg_count
    }
    return neg_count_dict, dep_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    saves_dir = os.path.join("saves", "negation_dep_parsing")
    Path(saves_dir).mkdir(parents=True, exist_ok=True) 

    ## Required parameters
    parser.add_argument("--pos_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--neg_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--name",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        help="")
    
    args = parser.parse_args()

    dep_data = []
    seed_val = 23
    np.random.seed(seed_val)

    for filename in [args.pos_file, args.neg_file]:
        texts = util.read_file(filename)

    selected_texts = util.get_samples(texts, args.n_samples)

    neg_count_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(neg_count_data.keys())
    for text in selected_texts:
        # print("Text: ", text)
        neg_count_dict, dep_dict = count_negation(text)
        for a_type in analysis_types:
            neg_count_data[a_type].append(neg_count_dict[a_type])
        dep_data.append({
            "text": text,
            "dep_info": dep_dict
            })
    for a_type in analysis_types:
        print(a_type, np.mean(neg_count_data[a_type]))
    pickle.dump(dep_data, open(os.path.join(saves_dir, args.name+"_dep_data.pickle"), "wb"))