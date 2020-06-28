#!/bin/sh
python code/compute_pvalue_ind.py --pickle_file=saves/liwc/all/liwc_dist_data.pickle
python code/compute_pvalue_ind.py --pickle_file=saves/pos_neg_dist/vader_pos_neg_dist.pickle
python code/compute_pvalue_ind.py --pickle_file=saves/pos_neg_negation/vader_pos_neg_negation_dist.pickle
python code/compute_pvalue_ind.py --pickle_file=saves/negation_dep_parsing_2_hops/all/pos_neg_negation_depparsing.pickle
python code/compute_pvalue_ind.py --pickle_file=saves/negation/vader_negation_only_dist.pickle
python code/compute_pvalue_ind.py --pickle_file=saves/negation_dep_parsing_2_hops/all/overall.pickle
python code/compute_pvalue_ind.py --pickle_file=saves/dataset_length/dataset_length_dist.pickle



python code/compute_pvalue_rel.py --pickle_file=saves/liwc/all/liwc_dist_data.pickle
python code/compute_pvalue_rel.py --pickle_file=saves/pos_neg_dist/vader_pos_neg_dist.pickle
python code/compute_pvalue_rel.py --pickle_file=saves/pos_neg_negation/vader_pos_neg_negation_dist.pickle
python code/compute_pvalue_rel.py --pickle_file=saves/negation_dep_parsing_2_hops/all/pos_neg_negation_depparsing.pickle