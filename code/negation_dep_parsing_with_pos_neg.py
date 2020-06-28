import spacy
import argparse
import numpy as np
import util
import pickle
import os
from pathlib import Path
import vader_negation_util

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
                if token.head.text in vader_sentiment_scores:
                    sent_score = vader_sentiment_scores[token.head.text]
                    if sent_score>=1:
                        pos_negation_count +=1 
                    elif sent_score<=-1:
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

    vader_sentiment_scores = vader_negation_util.read_vader_sentiment_dict(VADER_LEXICON_PATH)

    for filename in [args.pos_file, args.neg_file]:
        texts = util.read_file(filename)

    selected_texts = util.get_samples(texts, args.n_samples)

    negation_count_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }

    pos_negation_count_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }

    neg_negation_count_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }

    analysis_types = list(negation_count_data.keys())
    for text in selected_texts:
        # print("Text: ", text)
        negation_count_dict, pos_negation_count_dict, neg_negation_count_dict, dep_dict = count_negation(text, 
            vader_sentiment_scores)

        for a_type in analysis_types:
            negation_count_data[a_type].append(negation_count_dict[a_type])
            pos_negation_count_data[a_type].append(pos_negation_count_dict[a_type])
            neg_negation_count_data[a_type].append(neg_negation_count_dict[a_type])

        dep_data.append({
            "text": text,
            "dep_info": dep_dict
            })
    print("\nOverall negation dist.")
    for a_type in analysis_types:
        print(a_type, np.mean(negation_count_data[a_type]))

    print("\n\nPositive negation dist.")
    for a_type in analysis_types:
        print(a_type, np.mean(pos_negation_count_data[a_type]))

    print("\n\nNegative negation dist.")
    for a_type in analysis_types:
        print(a_type, np.mean(neg_negation_count_data[a_type]))

    pickle.dump(dep_data, open(os.path.join(saves_dir, args.name+"_dep_data.pickle"), "wb"))