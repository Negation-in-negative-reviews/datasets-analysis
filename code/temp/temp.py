# import spacy

# nlp = spacy.load("en_core_web_md")

# s = 'I was looking for to eating here for quite some time.  The place has a really nice old country style inn feel and the staff are friendly.  There is always a line up and it always looks like it is worth the wait.  We waited for about 15 minutes for a table and our expectations were extremely high.  The portions and plates are HUGE so make sure you come here hungover and super super hungry to get your moneys worth.  I ordered they usual eggs, bacon, french toast, and home fries.  Everything looked okay until I realized that everything tasted like salt.  There was salt in everything and it was hard to eat.  The french toast was horrible and it tasted like bread dipped in a really bad egg.  The eggs tasted like powdered eggs and were extremely hard to eat.  Overall, I left flips disappointed and will not go back.  Maybe they were having an "off" day or maybe my expectations were too high.... regardless... Flips was an utter fail for food.'

# doc = nlp(s)
# print(len(doc))
# sum = 0
# i = 0
# print(len(doc.sents))
# for sent in doc.sents:
#     i += 1
#     sent_doc = nlp(sent.text)
#     sum += len(sent_doc)

# print(sum)

def filter_util(x: list):
    if len(x)>0:
        return True
    else:
        return False

import pickle

data = pickle.load(open("pickle_saves/sst_neg_vader_negation.pickle", "rb"))

posemo_words_all = list(map(lambda x:x["positive_words"], data["negation_analysis_data"]))
posemo_words = list(filter(filter_util, posemo_words_all))
posemo_words_set = set()
for val1 in posemo_words:
    for v in val1:
        posemo_words_set.add(v)

negemo_words_set = set()
negemo_words_all = list(map(lambda x:x["negative_words"], data["negation_analysis_data"]))
negemo_words = list(filter(filter_util, negemo_words_all))
negemo_words_set = set()
for val1 in negemo_words:
    for v in val1:
        negemo_words_set.add(v)

# posemo_words_str = set(list(map(lambda x: " ".join(x), posemo_words)))

print(posemo_words_set)
print(negemo_words_set)
