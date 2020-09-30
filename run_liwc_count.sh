#!/bin/sh
nohup python code/liwc_count.py --datasets_info_json="./input.json" --liwc_filepath="/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic" &> nohup_liwc_count.out &