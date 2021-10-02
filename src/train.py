# src/train.py
import pandas as pd
import os
from sklearn import metrics
from sklearn import tree
import joblib
import argparse

import config
import model_dispatcher

def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    df.drop(columns=['location', 'id', 'initiation_drug', 'Unnamed: 0'], inplace=True)

    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from df
    # target is outcome column
    X_train = df_train.drop(columns=['outcome']).values
    y_train = df_train.outcome.values
    X_valid = df_valid.drop(columns=['outcome']).values
    y_valid = df_valid.outcome.values 

    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]

    # fit the model on the training data
    clf.fit(X_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(X_valid)

    # calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print("Fold = {}, Accuracy = {}".format(fold, accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()
    run(fold=args.fold, model=args.model)
