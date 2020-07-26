#!/bin/sh
nohup python code/negation_dep_parsing.py --analysis_type="word_level" &> nohup_negation_dep_parsing_word_level.out &
nohup python code/negation_dep_parsing.py --analysis_type="sent_level" &> nohup_negation_dep_parsing_sent_level.out &
nohup python code/negation_dep_parsing.py --analysis_type="review_level" &> nohup_negation_dep_parsing_review_level.out &