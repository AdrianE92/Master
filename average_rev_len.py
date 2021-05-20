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
cats = {}
games_train = []
games_dev = []
games_test = []
sports_train = []
sports_dev = []
sports_test = []
restaurants_train = []
restaurants_dev = []
restaurants_test = []
screen_train = []
screen_dev = []
screen_test = []
music_train = []
music_dev = []
music_test = []
misc_train = []
misc_dev = []
misc_test = []
products_train = []
products_dev = []
products_test = []
stage_train = []
stage_dev = []
stage_test = []
literature_train = []
literature_dev = []
literature_test = []

print('train')
for i in train:
    rev = pd.read_pickle(i)
    texts = rev['text']
    if i.name == "games.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in games_train:
                    games_train.append(k)
    elif i.name == "literature.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in literature_train:
                    literature_train.append(k)
    elif i.name == "misc.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in misc_train:
                    misc_train.append(k)
    elif i.name == "music.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in music_train:
                    music_train.append(k)
    elif i.name == "products.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in products_train:
                    products_train.append(k)
    elif i.name == "restaurants.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in restaurants_train:
                    restaurants_train.append(k)
    elif i.name == "screen.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in screen_train:
                    screen_train.append(k)
    elif i.name == "sports.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in sports_train:
                    sports_train.append(k)
    elif i.name == "stage.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in stage_train:
                    stage_train.append(k)
print('dev')
for i in dev:
    rev = pd.read_pickle(i)
    texts = rev['text']
    if i.name == "games.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in games_dev:
                    games_dev.append(k)
    elif i.name == "literature.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in literature_dev:
                    literature_dev.append(k)
    elif i.name == "misc.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in misc_dev:
                    misc_dev.append(k)
    elif i.name == "music.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in music_dev:
                    music_dev.append(k)
    elif i.name == "products.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in products_dev:
                    products_dev.append(k)
    elif i.name == "restaurants.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in restaurants_dev:
                    restaurants_dev.append(k)
    elif i.name == "screen.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in screen_dev:
                    screen_dev.append(k)
    elif i.name == "sports.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in sports_dev:
                    sports_dev.append(k)
    elif i.name == "stage.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in stage_dev:
                    stage_dev.append(k)
print('test')
for i in test:
    rev = pd.read_pickle(i)
    texts = rev['text']
    if i.name == "games.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in games_test:
                    games_test.append(k)
    elif i.name == "literature.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in literature_test:
                    literature_test.append(k)
    elif i.name == "misc.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in misc_test:
                    misc_test.append(k)
    elif i.name == "music.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in music_test:
                    music_test.append(k)
    elif i.name == "products.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in products_test:
                    products_test.append(k)
    elif i.name == "restaurants.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in restaurants_test:
                    restaurants_test.append(k)
    elif i.name == "screen.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in screen_test:
                    screen_test.append(k)
    elif i.name == "sports.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in sports_test:
                    sports_test.append(k)
    elif i.name == "stage.pkl":
        for j in texts:
            words = j.split(', ')
            for k in words:
                if k not in stage_test:
                    stage_test.append(k)


print("Unique Tokens")
print("Games train:", len(games_train))
print("Games dev:", len(games_dev))
print("Games test:", len(games_test))
print("Sports train:", len(sports_train))
print("Sports dev:", len(sports_dev))
print("Sports test:", len(sports_test))
print("Rest train:", len(restaurants_train))
print("Rest dev:", len(restaurants_dev))
print("Rest test:", len(restaurants_test))
print("Screen train:", len(screen_train))
print("Screen dev:", len(screen_dev))
print("Screen test:", len(screen_test))
print("Music train:", len(music_train))
print("Music dev:", len(music_dev))
print("Music test:", len(music_test))
print("Misc train:", len(misc_train))
print("Misc dev:", len(misc_dev))
print("Misc test:", len(misc_test))
print("Prod train:", len(products_train))
print("Prod dev:", len(products_dev))
print("Prod test:", len(products_test))
print("Stage train:", len(stage_train))
print("Stage dev:", len(stage_dev))
print("Stage test:", len(stage_test))
print("Lit train:", len(literature_train))
print("Lit dev:", len(literature_dev))
print("Lit test:", len(literature_test))

games_total = []
sports_total = []
rest_total = []
screen_total = []
music_total = []
misc_total = []
prod_total = []
stage_total = []
lit_total = []

print("Creating games_total")
for i in games_train:
    word = i.strip('[]')
    if word not in games_total:
        games_total.append(word)

for i in games_dev:
    word = i.strip('[]')
    if word not in games_total:
        games_total.append(word)

for i in games_test:
    word = i.strip('[]')
    if word not in games_total:
        games_total.append(word)
with open("games_total.txt", mode='w', encoding='utf-8') as f:
    for i in games_total:
        f.write("%s\n" % i)

print(len(games_total))

print("Creating sports_total")
for i in sports_train:
    word = i.strip('[]')
    if word not in sports_total:
        sports_total.append(word)

for i in sports_dev:
    word = i.strip('[]')
    if word not in sports_total:
        sports_total.append(word)

for i in sports_test:
    word = i.strip('[]')
    if word not in sports_total:
        sports_total.append(word)
with open("sports_total.txt", mode='w', encoding='utf-8') as f:
    for i in sports_total:
        f.write("%s\n" % i)
print(len(sports_total))

print("Creating rest_total")
for i in restaurants_train:
    word = i.strip('[]')
    if word not in rest_total:
        rest_total.append(word)

for i in restaurants_dev:
    word = i.strip('[]')
    if word not in rest_total:
        rest_total.append(word)

for i in restaurants_test:
    word = i.strip('[]')
    if word not in rest_total:
        rest_total.append(word)
with open("restuarants_total.txt", mode='w', encoding='utf-8') as f:
    for i in rest_total:
        f.write("%s\n" % i)
print(len(rest_total))

print("Creating screen_total")
for i in screen_train:
    word = i.strip('[]')
    if word not in screen_total:
        screen_total.append(word)

for i in screen_dev:
    word = i.strip('[]')
    if word not in screen_total:
        screen_total.append(word)

for i in screen_test:
    word = i.strip('[]')
    if word not in screen_total:
        screen_total.append(word)
with open("screen_total.txt", mode='w', encoding='utf-8') as f:
    for i in screen_total:
        f.write("%s\n" % i)
print(len(screen_total))

print("Creating music_total")
for i in music_train:
    word = i.strip('[]')
    if word not in music_total:
        music_total.append(word)

for i in music_dev:
    word = i.strip('[]')
    if word not in music_total:
        music_total.append(word)

for i in music_test:
    word = i.strip('[]')
    if word not in music_total:
        music_total.append(word)
with open("music_total.txt", mode='w', encoding='utf-8') as f:
    for i in music_total:
        f.write("%s\n" % i)
print(len(music_total))

print("Creating misc_total")
for i in misc_train:
    word = i.strip('[]')
    if word not in misc_total:
        misc_total.append(word)

for i in misc_dev:
    word = i.strip('[]')
    if word not in misc_total:
        misc_total.append(word)

for i in misc_test:
    word = i.strip('[]')
    if word not in misc_total:
        misc_total.append(word)
with open("misc_total.txt", mode='w', encoding='utf-8') as f:
    for i in misc_total:
        f.write("%s\n" % i)
print(len(misc_total))

print("Creating prod_total")
for i in products_train:
    word = i.strip('[]')
    if word not in prod_total:
        prod_total.append(word)

for i in products_dev:
    word = i.strip('[]')
    if word not in prod_total:
        prod_total.append(word)

for i in products_test:
    word = i.strip('[]')
    if word not in prod_total:
        prod_total.append(word)
with open("prod_total.txt", mode='w', encoding='utf-8') as f:
    for i in prod_total:
        f.write("%s\n" % i)
print(len(prod_total))

print("Creating stage_total")
for i in stage_train:
    word = i.strip('[]')
    if word not in stage_total:
        stage_total.append(word)

for i in stage_dev:
    word = i.strip('[]')
    if word not in stage_total:
        stage_total.append(word)

for i in stage_test:
    word = i.strip('[]')
    if word not in stage_total:
        stage_total.append(word)
with open("stage_total.txt", mode='w', encoding='utf-8') as f:
    for i in stage_total:
        f.write("%s\n" % i)
print(len(stage_total))

print("Creating lit_total")
for i in literature_train:
    word = i.strip('[]')
    if word not in lit_total:
        lit_total.append(word)

for i in literature_dev:
    word = i.strip('[]')
    if word not in lit_total:
        lit_total.append(word)

for i in literature_test:
    word = i.strip('[]')
    if word not in lit_total:
        lit_total.append(word)
with open("lit_total.txt", mode='w', encoding='utf-8') as f:
    for i in lit_total:
        f.write("%s\n" % i)
print(len(lit_total))

sports_stage = []
sports_screen = []
sports_music = []
sports_misc = []
sports_lit = []
sports_prod = []
sports_games = []
sports_rest = []

unique_stage = 0
unique_games = 0
unique_misc = 0
unique_music = 0
unique_lit = 0
unique_rest = 0
unique_prod = 0
unique_screen = 0

for i in sports_total:
    if i in stage_total:
        sports_stage.append(i)
        unique_stage += 1
    if i in screen_total:
        sports_screen.append(i)
        unique_screen += 1
    if i in games_total:
        sports_games.append(i)
        unique_games += 1
    if i in misc_total:
        sports_misc.append(i)
        unique_misc += 1
    if i in music_total:
        sports_music.append(i)
        unique_music += 1
    if i in lit_total:
        sports_lit.append(i)
        unique_lit += 1
    if i in rest_total:
        sports_rest.append(i)
        unique_rest += 1
    if i in prod_total:
        sports_prod.append(i)
        unique_prod += 1
print("Sports common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)
print("Misc:", unique_misc)
print("Lit:", unique_lit)
print("Prod:", unique_prod)
print("Games:", unique_games)
print("Rest:", unique_rest)
print("Stage:", unique_stage)
print("Creating sports vocabs")
with open("sports_stage_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_stage:
        f.write("%s\n" % i)

with open("sports_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_screen:
        f.write("%s\n" % i)

with open("sports_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_music:
        f.write("%s\n" % i)

with open("sports_misc_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_misc:
        f.write("%s\n" % i)

with open("sports_lit_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_lit:
        f.write("%s\n" % i)

with open("sports_prod_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_prod:
        f.write("%s\n" % i)

with open("sports_games_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_games:
        f.write("%s\n" % i)

with open("sports_rest_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in sports_rest:
        f.write("%s\n" % i)
print("Finished creating sports vocabs")
sports_total = []
sports_stage = []
sports_screen = []
sports_music = []
sports_misc = []
sports_lit = []
sports_prod = []
sports_games = []
sports_rest = []

unique_games = 0
unique_misc = 0
unique_music = 0
unique_lit = 0
unique_rest = 0
unique_prod = 0
unique_screen = 0

stage_screen = []
stage_music = []
stage_misc = []
stage_lit = []
stage_prod = []
stage_games = []
stage_rest = []
print("Creating misc vocab")
for i in stage_total:
    if i in screen_total:
        stage_screen.append(i)
        unique_screen += 1
    if i in games_total:
        stage_games.append(i)
        unique_games += 1
    if i in misc_total:
        stage_misc.append(i)
        unique_misc += 1
    if i in music_total:
        stage_music.append(i)
        unique_music += 1
    if i in lit_total:
        stage_lit.append(i)
        unique_lit += 1
    if i in rest_total:
        stage_rest.append(i)
        unique_rest += 1
    if i in prod_total:
        stage_prod.append(i)
        unique_prod += 1

print("Stage common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)
print("Misc:", unique_misc)
print("Lit:", unique_lit)
print("Prod:", unique_prod)
print("Games:", unique_games)
print("Rest:", unique_rest)

with open("stage_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_screen:
        f.write("%s\n" % i)

with open("stage_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_music:
        f.write("%s\n" % i)

with open("stage_misc_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_misc:
        f.write("%s\n" % i)

with open("stage_lit_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_lit:
        f.write("%s\n" % i)

with open("stage_prod_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_prod:
        f.write("%s\n" % i)

with open("stage_games_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_games:
        f.write("%s\n" % i)

with open("stage_rest_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in stage_rest:
        f.write("%s\n" % i)

print("Finished creating stage vocabs")

stage_total = []
stage_screen = []
stage_music = []
stage_misc = []
stage_lit = []
stage_prod = []
stage_games = []
stage_rest = []

unique_games = 0
unique_misc = 0
unique_music = 0
unique_lit = 0
unique_prod = 0
unique_screen = 0

rest_games = []
rest_misc = []
rest_music = []
rest_lit = []
rest_prod = []
rest_screen = []
print("Creating restaurant vocabs")
for i in rest_total:
    if i in screen_total:
        rest_screen.append(i)
        unique_screen += 1
    if i in games_total:
        rest_games.append(i)
        unique_games += 1
    if i in misc_total:
        rest_misc.append(i)
        unique_misc += 1
    if i in music_total:
        rest_music.append(i)
        unique_music += 1
    if i in lit_total:
        rest_lit.append(i)
        unique_lit += 1
    if i in prod_total:
        rest_prod.append(i)
        unique_prod += 1

print("Rest common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)
print("Misc:", unique_misc)
print("Lit:", unique_lit)
print("Prod:", unique_prod)
print("Games:", unique_games)
with open("rest_games_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in rest_games:
        f.write("%s\n" % i)

with open("rest_misc_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in rest_misc:
        f.write("%s\n" % i)

with open("rest_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in rest_music:
        f.write("%s\n" % i)

with open("rest_lit_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in rest_lit:
        f.write("%s\n" % i)

with open("rest_prod_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in rest_prod:
        f.write("%s\n" % i)

with open("rest_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in rest_screen:
        f.write("%s\n" % i)

print("Finished creating restaurant vocabs")
rest_total = []
rest_games = []
rest_misc = []
rest_music = []
rest_lit = []
rest_prod = []
rest_screen = []

unique_misc = 0
unique_music = 0
unique_lit = 0
unique_prod = 0
unique_screen = 0

games_misc = []
games_music = []
games_lit = []
games_prod = []
games_screen = []
print("Creating games vocabs")
for i in games_total:
    if i in screen_total:
        games_screen.append(i)
        unique_screen += 1
    if i in misc_total:
        games_misc.append(i)
        unique_misc += 1
    if i in music_total:
        games_music.append(i)
        unique_music += 1
    if i in lit_total:
        games_lit.append(i)
        unique_lit += 1
    if i in prod_total:
        games_prod.append(i)
        unique_prod += 1
print("Games common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)
print("Misc:", unique_misc)
print("Lit:", unique_lit)
print("Prod:", unique_prod)

with open("games_misc_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in games_misc:
        f.write("%s\n" % i)

with open("games_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in games_music:
        f.write("%s\n" % i)

with open("games_lit_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in games_lit:
        f.write("%s\n" % i)

with open("games_prod_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in games_prod:
        f.write("%s\n" % i)

with open("games_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in games_screen:
        f.write("%s\n" % i)

print("Finished creating games vocabs")
games_total = []
games_misc = []
games_music = []
games_lit = []
games_prod = []
games_screen = []

unique_misc = 0
unique_music = 0
unique_lit = 0
unique_screen = 0

prod_misc = []
prod_music = []
prod_lit = []
prod_screen = []

print("Creating products vocab")
for i in prod_total:
    if i in screen_total:
        prod_screen.append(i)
        unique_screen += 1
    if i in misc_total:
        prod_misc.append(i)
        unique_misc += 1
    if i in music_total:
        prod_music.append(i)
        unique_music += 1
    if i in lit_total:
        prod_lit.append(i)
        unique_lit += 1

print("Prod common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)
print("Misc:", unique_misc)
print("Lit:", unique_lit)


with open("prod_misc_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in prod_misc:
        f.write("%s\n" % i)

with open("prod_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in prod_music:
        f.write("%s\n" % i)

with open("prod_lit_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in prod_lit:
        f.write("%s\n" % i)

with open("prod_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in prod_screen:
        f.write("%s\n" % i)

print("Finished creating prod vocab")
prod_total = []
prod_misc = []
prod_music = []
prod_lit = []
prod_screen = []

unique_misc = 0
unique_music = 0
unique_screen = 0

lit_misc = []
lit_music = []
lit_screen = []
print("Creating literature vocabs")
for i in lit_total:
    if i in screen_total:
        lit_screen.append(i)
        unique_screen += 1
    if i in misc_total:
        lit_misc.append(i)
        unique_misc += 1
    if i in music_total:
        lit_music.append(i)
        unique_music += 1

print("Lit common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)
print("Misc:", unique_misc)

with open("lit_misc_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in lit_misc:
        f.write("%s\n" % i)

with open("lit_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in lit_music:
        f.write("%s\n" % i)

with open("lit_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in lit_screen:
        f.write("%s\n" % i)
print("Finished creating lit vocabs")

lit_total = []
lit_misc = []
lit_music = []
lit_screen = []

unique_music = 0
unique_screen = 0

misc_music = []
misc_screen = []
print("Creating misc vocabs")
for i in misc_total:
    if i in screen_total:
        misc_screen.append(i)
        unique_screen += 1
    if i in music_total:
        misc_music.append(i)
        unique_music += 1

print("Misc common tokens: ")
print("Screen:", unique_screen)
print("Music:", unique_music)

with open("misc_music_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in misc_music:
        f.write("%s\n" % i)

with open("misc_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in misc_screen:
        f.write("%s\n" % i)

print("Finished creating misc vocab")
misc_total = []
misc_music = []
misc_screen = []


unique_screen = 0
music_screen = []
print("Creating music vocab")
for i in music_total:
    if i in screen_total:
        music_screen.append(i)
        unique_screen += 1

print("Music common tokens: ")
print("Screen:", unique_screen)

with open("music_screen_vocab.txt", mode='w', encoding='utf-8') as f:
    for i in music_screen:
        f.write("%s\n" % i)
"""
for i in train:
    print(i.name)
    cats[i.name] = 0
    words = 0
    rev = pd.read_pickle(i)
    texts = rev['text']
    ratings = rev['rating']
    #print(ratings)
    for j in texts:
        words = j.split(',')
        for k in words:
            if k not in 
    #for j in ratings:
        #words += int(j)
    cats[i.name] += words/len(texts)

print('dev')
for i in dev:
    print(i)
    words = 0
    rev = pd.read_pickle(i)
    texts = rev['text']
    ratings = rev['rating']
    for j in texts:
        words += len(j.split(','))
    #for j in ratings:
        #words += int(j)
    cats[i.name] += words/len(texts)

print('test')
for i in test:
    print(i)
    words = 0
    rev = pd.read_pickle(i)
    texts = rev['text']
    ratings = rev['rating']
    for j in texts:
        words += len(j.split(','))
    #for j in ratings:
        #words += int(j)
    cats[i.name] += words/len(texts)

for i in cats:
    print(i)
    print(cats[i]/3)
"""