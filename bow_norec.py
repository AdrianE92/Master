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
import conllu
import matplotlib.pyplot as plt
import json
"""
TODO:
    Metadata:
    - Need to sort by domains

Clean up text (remove punctuations etc)
Remove stop words?
Sort by domains
Set grades as label
Train BoW for each domain and experiment

"""
if __name__ == "__main__":
    # Add command line arguments
    # This is probably the easiest way to store arguments for downstream
    parser = ArgumentParser()
    parser.add_argument(
        '--path', help="Path to the training corpus", action='store')
    parser.add_argument(
        '--meta', help="Path to the metadata", action='store')
    parser.add_argument('--vocab_size', help="How many words types to consider", action='store',
                        type=int, default=3000)
    parser.add_argument('--hidden_dim', help="Size of the hidden layer(s)", action='store',
                        type=int, default=64)
    parser.add_argument('--batch_size', help="Size of mini-batches", action='store', type=int,
                        default=16)
    parser.add_argument('--lr', action='store',
                        help="Learning rate", type=float, default=1e-3)
    parser.add_argument('--epochs', action='store', help="Max number of epochs", type=int,
                        default=15)
    parser.add_argument('--split', action='store', help="Ratio of train/dev split", type=float,
                        default=0.8)
    args = parser.parse_args()

    datafile = args.path
    metadata = args.meta
    # Set RNG seed for reproducibility
    torch.manual_seed(42)

    print('Loading the dataset...')

    with open(metadata, "r", encoding='utf-8') as f:
        data = f.read()
        meta = json.loads(data)



    train_dataset = pd.read_csv(
        datafile, sep='\t', header=0, compression='gzip')
    print('Finished loading the dataset')
    """
    #Uncomment this to reproduce plot of dataset before oversampling
    dist = train_dataset['source'].value_counts()
    dist.plot(kind='bar')
    plt.show()
    plt.close()
    """


    classes = train_dataset['source']  # Array with correct classes
    texts = train_dataset['text']

    # CountVectorizer creates a bag-of-words vector, using at most `max_features' words
    # binary = True indicates binary BoW
    # ngram_range set to (1,2) to include both unigrams and bigrams
    text_vectorizer = CountVectorizer(
        max_features=args.vocab_size, strip_accents='unicode', lowercase=False, binary=True, ngram_range=(1,2))
    # LabelEncoder returns integers for each label
    label_vectorizer = LabelEncoder()

    # We specify float32 because the default, float64, is inappropriate for pytorch:
    input_features = text_vectorizer.fit_transform(
        texts.values).toarray().astype(np.float32)
    gold_classes = label_vectorizer.fit_transform(classes.values)

    # Saving the vectorizer vocabulary for future usage (e.g., inference on test data):
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(text_vectorizer, f)

    print('Train data:', input_features.shape)

    # Number of classes:
    classes = label_vectorizer.classes_
    num_classes = len(classes)
    print(num_classes, 'classes')
    print(classes)

    # Splitting data with equal class distribution => stratify=gold_classes
    X_train, X_dev, y_train, y_dev = train_test_split(
        input_features, gold_classes, test_size=1-args.split, stratify=gold_classes)

    # Create a PyTorch object for the data:
    X_train, X_dev = torch.from_numpy(X_train).type(torch.float), torch.from_numpy(X_dev).type(torch.float)
    y_train, y_dev = torch.from_numpy(y_train).type(torch.long), torch.from_numpy(y_dev).type(torch.long)
    
    train = data.TensorDataset(X_train, y_train)
    dev = data.TensorDataset(X_dev, y_dev)
    print('Training instances after split:', len(train))
    
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