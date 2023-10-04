from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
import utils.visuals as visu
import numpy as np
import pandas as pd
import utils.processing as proc

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
visu.save_scatter_plots(iris_df, iris_df.corr())

X, y = iris.data, iris.target
# proc.split_data()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
y_predict = clf.predict(X)
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
proc.get_confusion_matrix(iris.target, y_predict, iris.target_names)

