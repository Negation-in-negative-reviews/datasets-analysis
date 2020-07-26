import pickle
import gzip
import json
import pprint
import argparse
import os

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--yelp_review_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--yelp_business_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    
    args = parser.parse_args() 
    myprint(f"args: {args}")   

    # yelp_review_filepath = "/data/madhu/yelp/yelp_academic_dataset_review.json.gz"
    # yelp_business_filepath = "/data/madhu/yelp/yelp_academic_dataset_business.json.gz"
    neg_review_filepath = os.path.join(args.output_dir, "review.0")
    pos_review_filepath = os.path.join(args.output_dir, "review.1")

    restaurant_business_ids = set()
    with gzip.open(args.yelp_business_filepath, "rb") as f:    
        jl_data = f.read().decode('utf-8') 
        jl_data = list(jl_data.split("\n"))
        for line in jl_data:  
            if line:    
                json_content = json.loads(line)
                if json_content["categories"] != None:                        
                    categories = [val.lower().strip() for val in json_content["categories"].split(",")]
                    if "restaurants" in categories:
                        restaurant_business_ids.add(json_content["business_id"])
    
    print("Finished reading the business.json file")
    neg_review_file = open(neg_review_filepath,"w")
    pos_review_file = open(pos_review_filepath,"w")
    with gzip.open(args.yelp_review_filepath, "rb") as f:    
        jl_data = f.read().decode('utf-8') 
        jl_data = list(jl_data.split("\n"))
        for line in jl_data:  
            if line:    
                json_content = json.loads(line)
                if json_content["business_id"] in restaurant_business_ids:
                    if float(json_content["stars"]) >= 3:
                        pos_review_file.write(json_content["text"].replace("\n", " ")+"\n")
                    elif float(json_content["stars"]) <= 2:
                        neg_review_file.write(json_content["text"].replace("\n", " ")+"\n")
                        
    print("Finished reading the review.json file")
