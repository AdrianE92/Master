import os
import matplotlib.pyplot as plt
import nltk
import json
import csv
import pandas as pd

#train = os.scandir("../norec/pre_proc_data/train/")
train = os.scandir("./data/train/")
dev = os.scandir("./data/dev/")
test = os.scandir("./data/test/")
"""
for i in train:
    for j in os.scandir(i):
        rev = pd.read_csv(j)
        #print(rev)

        print(rev.count(axis=0))
"""
print('train')
for i in train:
    print(i)
    words = 0
    rev = pd.read_pickle(i)
    texts = rev['text']
    for j in texts:
        words += len(j.split(','))
    print(words/len(texts))

print('dev')
for i in dev:
    print(i)
    words = 0
    rev = pd.read_pickle(i)
    texts = rev['text']
    for j in texts:
        words += len(j.split(','))
    print(words/len(texts))

print('test')
for i in test:
    print(i)
    words = 0
    rev = pd.read_pickle(i)
    texts = rev['text']
    for j in texts:
        words += len(j.split(','))
    print(words/len(texts))