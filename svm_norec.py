#!/bin/env python3
# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import json
if __name__ == "__main__":
    # Add command line arguments
    # This is probably the easiest way to store arguments for downstream
    parser = ArgumentParser()
    parser.add_argument(
        '--path', help="Path to the training corpus", action='store', default="data")
    parser.add_argument(
        '--train', help="Training category", action='store', default="screen")    
    parser.add_argument(
        '--dev', help="Dev category", action='store', default="screen")
    parser.add_argument('--vocab_size', help="How many words types to consider", action='store',
                        type=int, default=10000)


    torch.manual_seed(42)
    args = parser.parse_args()
    datafile = args.path
    cats = ['sports', 'games', 'music', 'misc', 'stage', 'literature', 'restaurants', 'products', 'screen']



    for i in cats:
        ratings = pd.Series()
        texts = pd.Series()
        for k in cats:
            train = pd.read_pickle(args.path + "/train/" + k + ".pkl")
            ratings = ratings.append(train['rating'])
            texts = texts.append(train['text'])
            print(len(texts))
        
        # Vectorize labels
        label_vectorizer = LabelEncoder()
        # Vectorize texts
        text_vectorizer = TfidfVectorizer()
        """
        text_vectorizer = CountVectorizer(
            max_features=args.vocab_size, strip_accents='unicode', lowercase=True, binary=False, ngram_range=(1,1))
        """
        # Transform texts
        input_features = text_vectorizer.fit_transform(
            texts.values)
        # Transform labels
        label_vectorizer.fit([1,2,3,4,5,6])
        gold_classes = label_vectorizer.transform(ratings.values)
        clf = svm.SVC(kernel='linear')
        clf.fit(input_features, gold_classes)
        for j in cats:
            print("Train", 'Everything')
            print("Test", j)
            
            dev = pd.read_pickle(args.path + "/dev/" + j + ".pkl")
            test = pd.read_pickle(args.path + "/test/" + j + ".pkl")
            # Set RNG seed for reproducibility

            # Split data in rating and text for train, dev and test.
            """
            ratings = train['rating']  
            texts = train['text']
            """

            dev_ratings = dev['rating']
            dev_texts = dev['text']

            test_ratings = test['rating']
            test_texts = test['text']
            

            
            # Transform dev_texts
            dev_input_features = text_vectorizer.transform(dev_texts).toarray()
            
            dev_gold_classes = label_vectorizer.transform(dev_ratings)
            
            # Transform test_texts
            test_input_features = text_vectorizer.transform(test_texts).toarray()
            
            # Transform test_labels
            test_gold_classes = label_vectorizer.transform(test_ratings)


            score = clf.score(test_input_features, test_gold_classes)
            clf = None
            print(score)
            
        break