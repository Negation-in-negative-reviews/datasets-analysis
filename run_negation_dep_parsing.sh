#!/bin/sh
nohup python code/negation_dep_parsing.py --analysis_type="word_level" --datasets_info_json="input.json" &> nohup_negation_depparsing_word_level.out &
nohup python code/negation_dep_parsing.py --analysis_type="sent_level" --datasets_info_json="input.json" &> nohup_negation_depparsing_sent_level.out &
nohup python code/negation_dep_parsing.py --analysis_type="review_level" --datasets_info_json="input.json" &> nohup_negation_depparsing_review_level.out &