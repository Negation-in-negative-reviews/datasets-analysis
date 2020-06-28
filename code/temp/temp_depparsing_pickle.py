import pickle

data = pickle.load(open("saves/negation_dep_parsing_2_hops/all/overall.pickle", "rb"))

new_data = {}
for key in ["review_level", "word_level", "sent_level"]:
    new_data[key] = []
    new_data[key] += data[key][2:4]
    new_data[key] += data[key][:2]
    new_data[key] += data[key][4:]

pickle.dump(new_data, open("saves/negation_dep_parsing_2_hops/all/overall_new.pickle", "wb"))



data = pickle.load(open("saves/negation_dep_parsing_2_hops/all/pos_neg_negation_depparsing.pickle", "rb"))

new_data = {}
for key in ["review_level", "word_level", "sent_level"]:
    new_data[key] = []
    new_data[key] += data[key][2*2:4*2]
    new_data[key] += data[key][:2*2]
    new_data[key] += data[key][4*2:]

pickle.dump(new_data, open("saves/negation_dep_parsing_2_hops/all/pos_neg_negation_depparsing_new.pickle", "wb"))