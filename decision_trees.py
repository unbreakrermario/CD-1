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
x_train_data = train_data[train_data.columns[0:4]]
y_train_data = train_data["target"]
x_test_data = test_data[test_data.columns[0:4]]
y_test_data = test_data["target"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train_data, y_train_data)
y_predict = clf.predict(x_test_data)  # x_test_data
iris.feature_names.remove("target")
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
confusion_matrix = proc.get_confusion_matrix(y_test_data, y_predict, iris.target_names)
visu.save_confusion_matrix(confusion_matrix)
