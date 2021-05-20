import os
import matplotlib.pyplot as plt
import nltk
import json
import csv
import pandas as pd

train = os.scandir("./data/train/")
dev = os.scandir("./data/dev/")
test = os.scandir("./data/test/")
games_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
sports_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
restaurants_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
screen_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
music_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
misc_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
products_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
stage_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
literature_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
screen_total = 0
music_total = 0
misc_total = 0
lit_total = 0
prod_total = 0
games_total = 0
rest_total = 0
stage_total = 0
sports_total = 0

for i in train:
    rev = pd.read_pickle(i)
    texts = rev['rating']
    if i.name == "games.pkl":
        for j in texts:
            games_train[j] += 1
            games_total += 1
    elif i.name == "literature.pkl":
        for j in texts:
            literature_train[j] += 1
            lit_total +=1
    elif i.name == "misc.pkl":
        for j in texts:
            misc_train[j] += 1
            misc_total += 1
    elif i.name == "music.pkl":
        for j in texts:
            music_train[j] += 1
            music_total +=1
    elif i.name == "products.pkl":
        for j in texts:
            products_train[j] += 1
            prod_total +=1
    elif i.name == "restaurants.pkl":
        for j in texts:
            restaurants_train[j] += 1
            rest_total += 1
    elif i.name == "screen.pkl":
        for j in texts:
            screen_train[j] += 1
            screen_total += 1
    elif i.name == "sports.pkl":
        for j in texts:
            sports_train[j] += 1
            sports_total += 1
    elif i.name == "stage.pkl":
        for j in texts:
            stage_train[j] += 1
            stage_total += 1

print("Screen train", screen_train, screen_total)
print("Music train", music_train, music_total)
print("Misc train", misc_train, misc_total)
print("Lit train", literature_train, lit_total)
print("Prod train", products_train, prod_total)
print("Games train", games_train, games_total)
print("Rest train", restaurants_train, rest_total)
print("Stage train", stage_train, stage_total)
print("Sports train", sports_train, sports_total)

games_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
sports_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
restaurants_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
screen_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
music_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
misc_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
products_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
stage_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
literature_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

for i in dev:
    rev = pd.read_pickle(i)
    texts = rev['rating']
    if i.name == "games.pkl":
        for j in texts:
            games_train[j] += 1
            games_total += 1
    elif i.name == "literature.pkl":
        for j in texts:
            literature_train[j] += 1
            lit_total += 1
    elif i.name == "misc.pkl":
        for j in texts:
            misc_train[j] += 1
            misc_total += 1
    elif i.name == "music.pkl":
        for j in texts:
            music_train[j] += 1
            music_total += 1
    elif i.name == "products.pkl":
        for j in texts:
            products_train[j] += 1
            prod_total += 1
    elif i.name == "restaurants.pkl":
        for j in texts:
            restaurants_train[j] += 1
            rest_total += 1
    elif i.name == "screen.pkl":
        for j in texts:
            screen_train[j] += 1
            screen_total += 1
    elif i.name == "sports.pkl":
        for j in texts:
            sports_train[j] += 1
            sports_total += 1
    elif i.name == "stage.pkl":
        for j in texts:
            stage_train[j] += 1
            stage_total += 1
print("-----------DEV--------------")
print("Screen train", screen_train, screen_total)
print("Music train", music_train, music_total)
print("Misc train", misc_train, misc_total)
print("Lit train", literature_train, lit_total)
print("Prod train", products_train, prod_total)
print("Games train", games_train, games_total)
print("Rest train", restaurants_train, rest_total)
print("Stage train", stage_train, stage_total)
print("Sports train", sports_train, sports_total)


games_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
sports_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
restaurants_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
screen_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
music_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
misc_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
products_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
stage_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
literature_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

for i in test:
    rev = pd.read_pickle(i)
    texts = rev['rating']
    if i.name == "games.pkl":
        for j in texts:
            games_train[j] += 1
            games_total += 1
    elif i.name == "literature.pkl":
        for j in texts:
            literature_train[j] += 1
            lit_total += 1
    elif i.name == "misc.pkl":
        for j in texts:
            misc_train[j] += 1
            misc_total += 1
    elif i.name == "music.pkl":
        for j in texts:
            music_train[j] += 1
            music_total += 1
    elif i.name == "products.pkl":
        for j in texts:
            products_train[j] += 1
            prod_total += 1
    elif i.name == "restaurants.pkl":
        for j in texts:
            restaurants_train[j] += 1
            rest_total += 1
    elif i.name == "screen.pkl":
        for j in texts:
            screen_train[j] += 1
            screen_total += 1
    elif i.name == "sports.pkl":
        for j in texts:
            sports_train[j] += 1
            sports_total += 1
    elif i.name == "stage.pkl":
        for j in texts:
            stage_train[j] += 1
            stage_total += 1
print("----------TEST---------------")
print("Screen train", screen_train, screen_total)
print("Music train", music_train, music_total)
print("Misc train", misc_train, misc_total)
print("Lit train", literature_train, lit_total)
print("Prod train", products_train, prod_total)
print("Games train", games_train, games_total)
print("Rest train", restaurants_train, rest_total)
print("Stage train", stage_train, stage_total)
print("Sports train", sports_train, sports_total)
