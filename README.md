Dataset Analysis
----

This repository contains code for lexicon analysis on Yelp, SST, IMDB movie reviews, Tripadvisor reviews and Amazon datasets. We use [LIWC](https://liwc.wpengine.com/), [Vader](https://github.com/cjhutto/vaderSentiment) for lexical operations and [Spacy](https://spacy.io/) for dependency parsing.

How to run
---
For each of the lexical analysis, we process the dataset and store the computed values in a pickle file. This pickle is used in jupyter notebooks to generate plots. Below are the list of different commands to be executed to precompute the lexicon values.

### Precomputing
1. Dataset length distribution
```
python code/dataset_length_dist.py --datasets_info_json="./input.json"
```
2. LIWC distribution
```
python code/liwc_count.py --datasets_info_json="./input.json" --liwc_filepath="/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic"
```
3. Lexicon distribution using Vader. You can find vader_lexicon.txt [here](https://github.com/cjhutto/vaderSentiment/blob/master/vaderSentiment/vader_lexicon.txt)
```
python code/vader_pos_neg_dist.py --datasets_info_json="./input.json" --vader_lexicon_path="/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"
```
3. Negation using LIWC
```
python code/liwc_negation.py --datasets_info_json="./input.json" --liwc_filepath="/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic"
```
4. Negation using Dependency Parsing
```
python code/negation_dep_parsing.py --analysis_type="word_level" --datasets_info_json="input.json"
python code/negation_dep_parsing.py --analysis_type="sent_level" --datasets_info_json="input.json"
python code/negation_dep_parsing.py --analysis_type="review_level" --datasets_info_json="input.json"
```
5. Negation using Vader lexicons
```
python code/vader_pos_neg_negation_dist.py --datasets_info_json="./input.json" --vader_lexicon_path="/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"
```

### Plotting

Run the following notebooks for generating the plots.

1. Dataset length distribution at sentence and review levels - `plot_dataset_length_dist.ipynb`
2. Lexicon distribution and negation distribution using LIWC - `plot_dist_using_liwc.ipynb`
3. Lexicon distribution using Vader - `plot_vader_pos_neg_dist.ipynb`
3. Negation distribution using Dependency parsing - `plot_negation_depparsing.ipynb`
4. Negation distribution using Vader lexicons - `plot_vader_negation.ipynb`