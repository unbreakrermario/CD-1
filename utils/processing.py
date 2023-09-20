import os
import math as mt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn import preprocessing


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


def normalize_diabetes_data(data):
    # formula ->  Z = (1/sqrt(n)) * ((x-mu)/std)
    mu_data = data.mean()
    std_data = data.std()
    normalized_data = data.sub(mu_data, axis='columns')
    normalized_data = normalized_data.div(std_data, axis='columns')
    val = (1 / mt.sqrt(442))
    normalized_data = normalized_data.mul(val, axis='columns')
    normalized_data["Y"] = data["Y"]
    return normalized_data


def split_data(data, split_percentage):
    training_data = data.sample(frac=split_percentage)
    idx_orig = data.index
    idx_train = training_data.index
    idx_test = idx_orig.drop(idx_train)
    test_data = data.loc[idx_test]
    return training_data, test_data


def simple_linear_regression(input_train, output_train):
    regr = linear_model.LinearRegression()
    inputs = input_train.values.reshape(-1, 3)
    regr.fit(inputs, output_train)
    return regr


def logistic_regression(input_train, output_train):
    model = linear_model.LogisticRegression()
    model.fit(input_train, output_train)
    return model


def test_predictions(model, input_test):
    output_test = model.predict(input_test.values.reshape(-1, 3))
    return output_test


def get_coefficients(model):
    return model.coef_


def get_mean_squared_error(output_test, output_predicted):
    return mean_squared_error(output_test, output_predicted)


def get_coefficient_determination(output_test, output_predicted):
    return r2_score(output_test, output_predicted)


def get_confusion_matrix(diabetes_y_test, diabetes_y_pred, target_names):
    cnf_matrix = confusion_matrix(diabetes_y_test, diabetes_y_pred)
    print(classification_report(diabetes_y_test, diabetes_y_pred, target_names=target_names))
    return cnf_matrix


def values_2_categorical(data):
    lab = preprocessing.LabelEncoder()
    return lab.fit_transform(data)

