import os
import pandas as pd


def check_output_folder(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
        print(path+" folder created")


def save_correlations(data):
    check_output_folder("output")
    data.to_csv("output/correlations.csv")


def get_correlations(data):
    correlations_data = data.corr()
    save_correlations(correlations_data)
    return correlations_data
