#!/bin/env python3
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.utils import data
import pickle
import numpy as np
#from helpers import eval_func, generate_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import json
#print(torch.cuda.is_available())
if __name__ == "__main__":
    # Add command line arguments
    # This is probably the easiest way to store arguments for downstream
    parser = ArgumentParser()
    parser.add_argument(
        '--path', help="Path to the training corpus", action='store', default="../norec/pre_proc_data/")
    parser.add_argument(
        '--train', help="Training category", action='store', default="sports")    
    parser.add_argument(
        '--dev', help="Dev category", action='store', default="sports")
    parser.add_argument('--vocab_size', help="How many words types to consider", action='store',
                        type=int, default=10000)
    parser.add_argument('--hidden_dim', help="Size of the hidden layer(s)", action='store',
                        type=int, default=256)
    parser.add_argument('--batch_size', help="Size of mini-batches", action='store', type=int,
                        default=16)
    parser.add_argument('--lr', action='store',
                        help="Learning rate", type=float, default=1e-3)
    parser.add_argument('--epochs', action='store', help="Max number of epochs", type=int,
                        default=15)

    args = parser.parse_args()
    datafile = args.path
    train = pd.read_pickle(args.path + "/train/" + args.train + ".pkl")
    dev = pd.read_pickle(args.path + "/dev/" + args.dev + ".pkl")
    # Set RNG seed for reproducibility
    torch.manual_seed(42)

    """
    print('Loading training data...')
    train_data = os.scandir(datafile + train)
    dfs = []
    for i in train_data:
        dfs.append(pd.read_csv(i, header=0))
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_pickle(datafile + train + "/" + args.train + ".pkl")
    frame = []

    print('Loading dev data...')
    dev_data = os.scandir(datafile + dev)
    dfs = []
    for i in dev_data:
        dfs.append(pd.read_csv(i, header=0))
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_pickle(datafile + dev + "/" + args.dev + ".pkl")
    print(frame)
    frame = []

    print('Loading test data...')
    test_data = os.scandir(datafile + test)
    dfs = []
    for i in test_data:
        dfs.append(pd.read_csv(i, header=0))
    frame = pd.concat(dfs, axis=0, ignore_index=True)
    frame.to_pickle(datafile + test + "/" + args.test + ".pkl")
    frame = []


    col1 = rating
    col2 = text
    row = review
        Rating    Text
            6   djsiaodjsaiod
            2   dsadsadsadsda
    """     


    ratings = train['rating']  # Array with correct ratings
    #print(ratings)
    texts = train['text']

    dev_ratings = dev['rating']
    dev_texts = dev['text']

    bow_ratings = ratings.append(dev_ratings)
    bow_texts = texts.append(dev_texts)
    # CountVectorizer creates a bag-of-words vector, using at most `max_features' words
    # binary = True indicates binary BoW
    # ngram_range set to (1,2) to include both unigrams and bigrams
    text_vectorizer = CountVectorizer(
        max_features=args.vocab_size, strip_accents='unicode', lowercase=False, binary=True, ngram_range=(1,2))
    
    # LabelEncoder returns integers for each label
    label_vectorizer = LabelEncoder()

    # We specify float32 because the default, float64, is inappropriate for pytorch:
    input_features = text_vectorizer.fit_transform(
        bow_texts.values).toarray().astype(np.float32)
    gold_classes = label_vectorizer.fit_transform(bow_ratings.values)
    """
    dev_input_features = text_vectorizer.fit_transform(dev_texts.values).toarray().astype(np.float32)
    dev_gold_classes = label_vectorizer.fit_transform(dev_ratings.values)
    """
    print(len(train), len(dev))
    print(len(bow_texts))
    print(input_features.shape)
    input_features, dev_input_features = input_features[len(train):, :], input_features[:len(train), :]
    gold_classes, dev_gold_classes = gold_classes[len(train):], gold_classes[:len(train)]
    # Saving the vectorizer vocabulary for future usage (e.g., inference on test data):
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(text_vectorizer, f)

    print('Train data:', input_features.shape)
    print('Dev data:', dev_input_features.shape)
    # Number of classes:
    classes = label_vectorizer.classes_
    num_classes = len(classes)
    """
    print(num_classes, 'classes')
    print(classes)
    # Splitting data with equal class distribution => stratify=gold_classes
    X_train, X_dev, y_train, y_dev = train_test_split(
        input_features, gold_classes, test_size=1-args.split, stratify=gold_classes)
    """
    
    # Create a PyTorch object for the data:
    X_features, X_classes = torch.from_numpy(input_features).type(torch.float), torch.from_numpy(gold_classes).type(torch.long)
    Y_features, Y_classes = torch.from_numpy(dev_input_features).type(torch.float), torch.from_numpy(dev_gold_classes).type(torch.long)
    #print(input_features, gold_classes)
    train = data.TensorDataset(X_features, X_classes)
    dev = data.TensorDataset(Y_features, Y_classes)
    #print('Training instances after split:', len(train))
    
    train_iter = data.DataLoader(   
        train, batch_size=args.batch_size, shuffle=True)
    dev_iter = data.DataLoader(dev, batch_size=len(dev), shuffle=False)

    # Implementing a model with 1 hidden layer and Stochastic Gradient Descent with momentum set to 9.
    model = nn.Sequential(nn.Linear(args.vocab_size, args.hidden_dim),
                          nn.ReLU(),
                          nn.Linear(args.hidden_dim, args.hidden_dim),
                          nn.ReLU(),
                          nn.Linear(args.hidden_dim, num_classes))
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        # Training
        for X_train, y_train in train_iter:
            y_pred = model(X_train)

            loss = loss_fn(y_pred, y_train)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print("epoch: {}\tloss: {}".format(epoch, loss.item()))

    # Evaluation
    # By now, you should have the arrays with gold labels of the development set (dev_labels),
    # and the predictions of your model on the same development set (dev_predictions).
    dev_labels = []
    dev_predictions = []
    
    for X_dev, y_dev in dev_iter:
        dev_labels.append(y_dev)
        pred = model(X_dev)
        pred = pred.max(dim=1)[1]
        dev_predictions.append(pred)
    dev_predictions = dev_predictions[0]
    dev_labels = dev_labels[0]

    dev_accuracy = accuracy_score(dev_labels, dev_predictions)
    print("Accuracy on the dev set:", dev_accuracy)
    
    print('Classification report for the dev set:')
    gold_classes_human = [classes[x] for x in dev_labels]
    predicted_dev_human = [classes[x] for x in dev_predictions]
    print(classification_report(gold_classes_human, predicted_dev_human))
    
    # Saving model
    torch.save(model, 'bow_model.pt')