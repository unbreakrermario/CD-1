import pandas as pd


def read_diabetes_dataset(path):
    diabetes_dataset = pd.read_csv(path, sep='\t')
    return diabetes_dataset


def read_natalidad_dataset_2022(path):
    diabetes_dataset = pd.read_csv(path, sep=',')
    return diabetes_dataset
