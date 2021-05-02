import os
import pandas as pd


def load_data():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    train = pd.read_csv(dir_name + '/train.tsv', sep='\t', index_col=0)
    train_x = train.drop('Type', axis=1)
    train_y = train['Type']
    test_x = pd.read_csv(dir_name + '/test.tsv', sep='\t', index_col=0)
    return train_x, train_y, test_x
