import argparse
from pathlib import Path
import vader_negation_util
import spacy
import torch
import os
import sys
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import random
from pathlib import Path
import util
import datetime
import logging
import pprint
import pandas as pd
import argparse
from vader_negation_util import NEGATE
import liwc_util

nlp = spacy.load("en_core_web_md")
VADER_LEXICON_PATH = "/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"

def get_sentence_sentiment(args: dict(), texts):
    seed_val = args.seed_val
    device = util.get_device(device_no=args.device_no)   
    model = torch.load(args.model_path, map_location=device)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # testfile = args.input_file
    # true_label = args.label
    truncation = args.truncation
    # n_samples = None
    # if "n_samples" in args:
    #     n_samples = args.n_samples
    
    # Load the BERT tokenizer.
    # logger.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0
    reviews = []
    # labels = []
    # with open(testfile, "r") as fin:
    #     reviews = fin.readlines()
    
    # reviews = [rev.lower() for rev in reviews]
    
    # if n_samples == None:
    #     n_samples = len(reviews)

    # indices = np.random.choice(np.arange(len(reviews)), size=n_samples)
    # selected_reviews = [reviews[idx] for idx in indices]

    # labels = [0 if true_label == "negative" else 1]*len(selected_reviews)
    # For every sentence...
    # for rev in selected_reviews:
    #     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    #     input_ids = tokenizer.encode(rev, add_special_tokens=True)
    #     # Update the maximum sentence length.
    #     max_len = max(max_len, len(input_ids))
        
    # print('Max sentence length: ', max_len)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for rev in texts:
        input_id = tokenizer.encode(rev, add_special_tokens=True)
        if len(input_id) > 512:                        
            if truncation == "tail-only":
                input_id = [tokenizer.cls_token_id]+input_id[-511:]      
            elif truncation == "head-and-tail":
                input_id = [tokenizer.cls_token_id]+input_id[1:129]+input_id[-382:]+[tokenizer.sep_token_id]
            else:                
                input_id = input_id[:511]+[tokenizer.sep_token_id]
                
            input_ids.append(torch.tensor(input_id).view(1,-1))
            attention_masks.append(torch.ones([1,len(input_id)], dtype=torch.long))
        else:
            encoded_dict = tokenizer.encode_plus(
                                rev,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels)

    # Set the batch size.  
    batch_size = 8  

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    print('Predicting labels for {:,} sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)
    
    print('    DONE.')
    # return predictions, true_labels

    # preds, true_labels = test(args, False, seed_val)
    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    return flat_predictions


# def mark_sample(text, vader_sentiment_scores):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--latex",
                        action='store_true',
                        help="")
    parser.add_argument("--liwc",
                        action='store_true',
                        help="")
    parser.add_argument("--truncation",
                        default="head-and-tail",
                        type=str,
                        # required=True,
                        help="")
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        help="")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    parser.add_argument("--device_no",
                        default=2,
                        type=int,
                        help="")
    parser.add_argument("--filter_threshold",
                        default=-1,
                        type=int,
                        help="")
    
    args = parser.parse_args()
    vader_sentiment_scores = vader_negation_util.read_vader_sentiment_dict(VADER_LEXICON_PATH)
    texts = util.read_file(args.input_file)
    if args.filter_threshold != -1:
        texts = util.filter_samples(texts, args.filter_threshold)

    samples = util.get_samples(texts, args.n_samples, args.seed_val)
    # samples = [s.lower() for s in samples]
    # print("Samples: ")
    # print("---------------")
    # for s in samples:
    #     print(s)
    # print("\n\n")
    processed_texts = []
    sample_sentences = []
    # for txt in samples:
    #     output_text = ""
    #     doc = nlp(txt)
    #     for sent in doc.sents:
    #         sample_sentences.append(sent.text)
    
    # preds = get_sentence_sentiment(args, sample_sentences)
    # print(len(preds))
    count = 0
    neg_words = []
    neg_words.extend(NEGATE)
    if args.liwc:
        liwc_filepath = "/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic"
        liwc_result, liwc_class_id, liwc_cluster_result, liwc_categories, liwc_category_reverse = liwc_util.load_liwc(liwc_filepath)            

    for txt in samples:        
        doc = nlp(txt)
        output_text = ""
        for sent in doc.sents:            
            doc_sent = nlp(sent.text)
            sent_text = ""
            words = sent.text.strip("\n").split()
            # negation_words = vader_negation_util.negated_returns_words(words)            
            for token in doc_sent:   
                token_proc_text = ""
                token_text = token.text.lower()
                if args.liwc:
                    if token_text in liwc_result:
                        liwc_cats_temp = [liwc_categories[val] for val in liwc_result[token_text]]
                        if "posemo" in liwc_cats_temp:
                            token_proc_text = "\\textcolor{blue}{"+token.text+"} "
                        elif "negemo" in liwc_cats_temp:
                            token_proc_text = "\\textcolor{red}{"+token.text+"} "
                        else:
                            token_proc_text = token.text
                    else:
                        token_proc_text = token.text
                else:             
                    if token_text in vader_sentiment_scores:
                        sent_score = vader_sentiment_scores[token_text]
                        if sent_score >= 1:
                            if args.latex:
                                token_proc_text = "\\textcolor{blue}{"+token.text+"} "
                            else:
                                token_proc_text = token.text+"<pos> "
                        elif sent_score <= -1:
                            if args.latex:
                                token_proc_text = "\\textcolor{red}{"+token.text+"} "
                            else:
                                token_proc_text = token.text+"<neg> "
                        else:
                            token_proc_text = token.text
                    else:
                        token_proc_text = token.text

                
                if token_text in neg_words or "n't" in token_text:
                    token_proc_text = "\\textbf{"+token_proc_text+"}"
                sent_text += token_proc_text+" "

            # if preds[count] == 1:
            #     output_text += "\\textbf{"+sent_text+"}"                
            # elif preds[count] == 0:
            #     output_text += sent_text
            # count += 1
            output_text += sent_text
            
        processed_texts.append(output_text.strip())

    fout = open(args.output_file, "w")

    print("Processed texts: ")
    print("===============")
    for text, proc_text in zip(samples, processed_texts):
        print("text:", text)
        print("processed text:", proc_text)
        # fout.write("text: "+text+"\n"+"")
        fout.write(("\item "+proc_text.replace("$", "\$").replace("%", "\%")).strip("\n")+"\n")
        # print()
        # print("------------------")







    