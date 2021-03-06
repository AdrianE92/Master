import os
import matplotlib.pyplot as plt
import nltk
import json
import csv
import pandas as pd
"""
TODO:
    Metadata:
    - Need to sort by domains

Clean up text (remove punctuations etc)
Sort by domains
Set grades as label
Train BoW for each domain and experiment

"""

#Create new folder with new text
"""
Train   - Dom1  - 0001
                - 0002
                - ...
        - Dom2  - 0003
                - 0004
                - ...

Dev
Test



"""
nor_tokenizer = nltk.data.load('tokenizers/punkt/norwegian.pickle')
train = os.scandir("../norec/data//train/")
dev = os.scandir("../norec/data/dev/")
test = os.scandir("../norec/data/test/")
#test = "./data/test/sports.pkl"
metadata = "../norec/data/metadata.json"
train_rating = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
train_lng = {'nb': 0, 'nn': 0}
train_total = 0
header = ["rating", "text"]
with open(metadata, "r", encoding='utf-8') as f:
    data = f.read()
    meta = json.loads(data)
cats = {}
sources = {}
"""
data = pd.read_pickle(test)
data.columns = ['rating', 'text']
texts = data.text.to_list()
texts2 = [""] * len(texts)
print(texts)
for i in range(len(texts)):
    counter = 0
    tmp = texts[i].split(', ')
    for j in tmp:
        counter += 1
        if counter == 256:
            continue
        word = j.strip("[]'")
        texts2[i] = texts2[i] + word + " "
    #print(texts2[i])

print(texts2)
"""
"""
for i in meta:
    #print(meta[i]["category"])
    if not meta[i]["category"] in cats:
        cats[meta[i]["category"]] = 0
    else:
        cats[meta[i]["category"]] += 1

for i in meta:
    train_rating[meta[i]["rating"]] += 1
    train_lng[meta[i]["language"]] += 1

for i in meta:
    #print(meta[i]["category"])
    if not meta[i]["source"] in sources:
        sources[meta[i]["source"]] = 0
    else:
        sources[meta[i]["source"]] += 1
Category: Rating
Total: Rating
"""
#print(cats)
for i in train:
    category = meta[i.name[:-4]]["category"]
    rating = meta[i.name[:-4]]["rating"]
    with open(i, "r", encoding='utf-8') as f:
        data = f.read()
        toks = nltk.word_tokenize(data, language="norwegian")
        #toks = [j.lower() for j in toks]
        toks = [toks]
        toks.insert(0, str(rating))


    with open("../norec/pad/train/" + category + "/" + i.name[:-4] + ".csv", mode="w", encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerow(toks)

print("train complete")

for i in test:
    category = meta[i.name[:-4]]["category"]
    rating = meta[i.name[:-4]]["rating"]
    with open(i, "r", encoding='utf-8') as f:
        data = f.read()
        toks = nltk.word_tokenize(data, language="norwegian")
        #toks = [j.lower() for j in toks]
        toks = [toks]
        toks.insert(0, str(rating))

    with open("../norec/pad/test/" + category + "/" + i.name[:-4] + ".csv", mode="w", encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerow(toks)

print("test complete")

for i in dev:
    category = meta[i.name[:-4]]["category"]
    rating = meta[i.name[:-4]]["rating"]
    with open(i, "r", encoding='utf-8') as f:
        data = f.read()
        toks = nltk.word_tokenize(data, language="norwegian")
        #toks = [j.lower() for j in toks]
        toks = [toks]
        toks.insert(0, str(rating))

    with open("../norec/pad/dev/" + category + "/" + i.name[:-4] + ".csv", mode="w", encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerow(toks)
print("dev complete")
"""
path = "../norec/pre_proc_data/train/games/000378.csv"
data = pd.read_csv(path, header=0)
lngs = ["nb", "nn"]
ratings = ["1", "2", "3", "4", "5", "6"]
source = [i for i in sources.keys()]
source_values = [i for i in sources.values()]
rt_values = [i for i in train_rating.values()]
lng_values = [i for i in train_lng.values()]
print(rt_values)
print(lng_values)
print(train_total)
"""

"""
tot = 28158
train = 353, 1973, 5061, 9174, 9905, 1692
train = 27752, 406

tot = 3518
dev = 44, 186, 588, 1131, 1343, 226
dev = 3461, 57

tot = 3513
test = 25, 206, 576, 1147, 1355, 204
test = 3443, 70

plt.bar(source, source_values)
#plt.bar(ratings, rt_values)
#plt.bar(lngs, lng_values)
plt.show()
plt.close()
"""