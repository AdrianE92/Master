import os
import matplotlib.pyplot as plt
import nltk
import json
import csv
import pandas as pd

train = os.scandir("./data/train/")
dev = os.scandir("./data/dev/")
test = os.scandir("./data/test/")

lit = {}
screen = {}
misc = {}
music = {}
prod = {}

print('train')
for i in train:
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