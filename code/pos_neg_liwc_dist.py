import liwc_util
import re
import pickle

if __name__=="__main__":
    '''
        result: a dictionary that maps each word to the LIWC cluster ids
            that it belongs to
        class_id: a dict that maps LIWC cluster to category id,
            this does not seem useful, here for legacy reasons
        cluster_result: a dict that maps LIWC cluster id to all words
            in that cluster
        categories: a dict that maps LIWC cluster to its name
        category_reverse: a dict that maps LIWC cluster name to its id
    '''
    liwc_filepath = "/data/LIWC2007/Dictionaries/LIWC2001_English.dic"
    data_filepath = "data/yelp/sentiment.train.0"
    pickle_save_filepath = "pickle_saves/clustercount_yelp_train0.pickle"
    result, class_id, cluster_result, categories, category_reverse = liwc_util.load_liwc(liwc_filepath)

    cluster_count = {}
    i=0
    with open(data_filepath, "r") as f:
        for sent in f.readlines():
            print(i)
            sent = sent.strip('\n')
            for word in sent.split():
                for pattern in result:
                    if (pattern.endswith("*") and word.startswith(pattern[:-1])) or (pattern==word):
                        for c in result[pattern]:
                            cluster_count[categories[c]] = cluster_count.get(categories[c], 0)+1
            i += 1

    pickle.dump(cluster_count, open(pickle_save_filepath, "wb"))

    print()