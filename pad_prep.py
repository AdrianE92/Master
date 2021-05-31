import os
import matplotlib.pyplot as plt
import nltk
import json
import csv
import pandas as pd
import stanza
stanza.download('no')
nlp = stanza.Pipeline(lang='no', processors='tokenize')
train = os.scandir("./pad/train/")
dev = os.scandir("./pad/dev/")
test = os.scandir("./pad/test/")

lit = {}
screen = {}
misc = {}
music = {}
prod = {}
datafile = "../norec/pad/"
print('Loading training data...')
cats = ["screen", "games", "sports", "restaurants", "music", "literature", "products", "stage", "misc"]
"""
for j in cats:
    dfs = []
    train_data = os.scandir(datafile + "train/" + j + "/")
    pkl_file = datafile + "train/" + j + "/"
    for i in train_data:
        #print(i)
        dfs.append(pd.read_csv(i, header=0))
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_pickle(pkl_file + j + ".pkl")
    frame = []
print('Loading dev data...')
#dev_data = os.scandir(datafile + dev)
for j in cats:
    dfs = []
    dev_data = os.scandir(datafile + "dev/" + j + "/")
    pkl_file = datafile + "dev/" + j + "/"
    for i in dev_data:
        dfs.append(pd.read_csv(i, header=0))
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_pickle(pkl_file + j + ".pkl")
    #print(frame)
    frame = []

print('Loading test data...')
#test_data = os.scandir(datafile + test)
for j in cats:
    dfs = []
    test_data = os.scandir(datafile + "test/" + j + "/")
    pkl_file = datafile + "test/" + j + "/"
    for i in test_data:
        dfs.append(pd.read_csv(i, header=0))
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_pickle(pkl_file + j + ".pkl")
    frame = []

"""
print('train')
for i in train:
    rev = pd.read_pickle(i)
    texts = rev['text']
    if i.name == "literature.pkl":
        for j in texts:
            lit = ""
            words = j.split(', ')
            for k in words:
                word = k.strip("[]'")
                if word in [".", ",", ":", "!", ";"]:
                    lit = lit[:-1] + word + " "
                else:
                    lit = lit + word + " "
            """
            print(lit)
            """
            doc = nlp(lit)
            print(lit)
            #print([sent.text for sent in doc.sentences])
            for i, sentence in enumerate(doc.sentences):
                print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
            """
            #print([sentence.text for sentence in doc.sentences])
            for i in doc.sentences:
                print(i.text)
            """
            exit()
                
    elif i.name == "misc.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in misc:
                    misc[word] = 1
                else:
                    misc[word] += 1

    elif i.name == "music.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in music:
                    music[word] = 1
                else:
                    music[word] += 1
    elif i.name == "products.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in prod:
                    prod[word] = 1
                else:
                    prod[word] += 1
    elif i.name == "screen.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in screen:
                    screen[word] = 1
                else:
                    screen[word] += 1

print('dev')
for i in dev:
    rev = pd.read_pickle(i)
    texts = rev['text']
    if i.name == "literature.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in lit:
                    lit[word] = 1
                else:
                    lit[word] += 1

    elif i.name == "misc.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in misc:
                    misc[word] = 1
                else:
                    misc[word] += 1

    elif i.name == "music.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in music:
                    music[word] = 1
                else:
                    music[word] += 1
    elif i.name == "products.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in prod:
                    prod[word] = 1
                else:
                    prod[word] += 1
    elif i.name == "screen.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in screen:
                    screen[word] = 1
                else:
                    screen[word] += 1

print('test')
for i in test:
    rev = pd.read_pickle(i)
    texts = rev['text']
    if i.name == "literature.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in lit:
                    lit[word] = 1
                else:
                    lit[word] += 1

    elif i.name == "misc.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in misc:
                    misc[word] = 1
                else:
                    misc[word] += 1

    elif i.name == "music.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in music:
                    music[word] = 1
                else:
                    music[word] += 1
    elif i.name == "products.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in prod:
                    prod[word] = 1
                else:
                    prod[word] += 1
    elif i.name == "screen.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                word = k.strip('[]')
                if word not in screen:
                    screen[word] = 1
                else:
                    screen[word] += 1

sort_screen = sorted(screen, key=screen.get, reverse=True)
sort_misc = sorted(misc, key=misc.get, reverse=True)
sort_music = sorted(music, key=music.get, reverse=True)
sort_prod = sorted(prod, key=prod.get, reverse=True)
sort_lit = sorted(lit, key=lit.get, reverse=True)
sort_screen = sort_screen[:55000]
sort_misc = sort_misc[:55000]
sort_music = sort_music[:55000]
sort_prod = sort_prod[:55000]
sort_lit = sort_lit[:55000]

with open("lit_total.txt", mode='w', encoding='utf-8') as f:
    for i in sort_lit:
        f.write("%s\n" % i)
with open("prod_total.txt", mode='w', encoding='utf-8') as f:
    for i in sort_prod:
        f.write("%s\n" % i)
with open("misc_total.txt", mode='w', encoding='utf-8') as f:
    for i in sort_misc:
        f.write("%s\n" % i)
with open("music_total.txt", mode='w', encoding='utf-8') as f:
    for i in sort_music:
        f.write("%s\n" % i)
with open("screen_total.txt", mode='w', encoding='utf-8') as f:
    for i in sort_screen:
        f.write("%s\n" % i)

"""
screen = sort_screen.keys()
misc = sort_misc.keys()
music = sort_music.keys()
prod = sort_prod.keys()
lit = sort_lit.keys()
"""