import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df_train = pd.read_csv("/Users/charlesoneill/diabeticretinopathy/input/df_train_one.csv")
    df_test = pd.read_csv("/Users/charlesoneill/diabeticretinopathy/input/df_test_one.csv")
    df = pd.concat([df_train, df_test])

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomise the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv("df_one_folds.csv", index=False)