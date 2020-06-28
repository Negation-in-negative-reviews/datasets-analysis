#!/bin/sh

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/stanford-sentiment-treebank/processed_data/pos_reviews_train --neg_file=/data/madhu/stanford-sentiment-treebank/processed_data/neg_reviews_train --name="sst" &> nohup_negation_dep_parsing_sst.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/imdb_dataset/processed_data/pos_reviews_train --neg_file=/data/madhu/imdb_dataset/processed_data/neg_reviews_train --n_samples=5000 --name="imdb" &> nohup_negation_dep_parsing_imdb.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/yelp/processed_data/review.1 --neg_file=/data/madhu/yelp/processed_data/review.0 --n_samples=5000 --name="yelp" &> nohup_negation_dep_parsing_yelp.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/tripadvisor/processed_data/pos_reviews_train --neg_file=/data/madhu/tripadvisor/processed_data/neg_reviews_train --n_samples=5000 --name="tripadvisor" &> nohup_negation_dep_parsing_tripadvisor.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/amazon-reviews-2018/processed_data/automotive/pos_reviews_train --neg_file=/data/madhu/amazon-reviews-2018/processed_data/automotive/neg_reviews_train --n_samples=5000 --name="automotive" &> nohup_negation_dep_parsing_automotive.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/pos_reviews_train --neg_file=/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/neg_reviews_train --n_samples=5000 --name="cellphones" &> nohup_negation_dep_parsing_cellphones.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/pos_reviews_train --neg_file=/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/neg_reviews_train --n_samples=5000 --name="luxury_beauty" &> nohup_negation_dep_parsing_luxury_beauty.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/pos_reviews_train --neg_file=/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/neg_reviews_train --n_samples=5000 --name="pet_supplies" &> nohup_negation_dep_parsing_pet_supplies.out &

nohup python code/negation_dep_parsing_with_pos_neg.py --pos_file=/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/pos_reviews_train --neg_file=/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/neg_reviews_train --n_samples=5000 --name="sports_and_outdoors" &> nohup_negation_dep_parsing_sports_and_outdoors.out &