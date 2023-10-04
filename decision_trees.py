from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
import utils.visuals as visu
import pandas as pd
import utils.processing as proc
import numpy as np

iris = load_iris()
y = np.expand_dims(iris.target, axis=1)
irisArray = np.concatenate((iris.data, y), axis=1)
iris.feature_names.append("target")
iris_df = pd.DataFrame(irisArray, columns=iris.feature_names)
visu.save_scatter_plots(iris_df, iris_df.corr())

train_data, test_data = proc.split_data(iris_df, 0.7)
# train_data split --> x_train_data, y_train_data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
y_predict = clf.predict(X) # x_test_data
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
proc.get_confusion_matrix(iris.target, y_predict, iris.target_names)

