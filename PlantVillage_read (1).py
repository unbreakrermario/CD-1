import os
import numpy as np
import cv2
import pandas as pd


def load_images_from_folder(path_general, folder, folders):
    images = pd.DataFrame(columns=range(128*128))
    y = pd.DataFrame(columns=list('Y'))
    for filename in os.listdir(path_general+folder):
        if any([filename.endswith(x) for x in ['.JPG', '.jpg']]):
            img = cv2.imread(os.path.join(path_general+folder, filename))
            if img is not None:
                img = img[:, :, 0]
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
                y.loc[len(y.index)] = [folders.index(folder)]
                images.loc[len(images.index)] = img.flatten()
                cv2.imshow("images", img)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit


    return y, images

folders = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]
y_out = pd.DataFrame(columns=list('Y'))
x_out = pd.DataFrame(columns=range(128*128))
for folder in folders:
    outY, outX = load_images_from_folder("E:/datasets/PlantVillage-Dataset/raw/grayscale/", folder, folders)
    y_out = pd.concat([y_out, outY])
    x_out = pd.concat([x_out, outX])
cv2.destroyAllWindows()
y_out = y_out.reset_index(drop=True)
x_out = x_out.reset_index(drop=True)
print(y_out.tail())

# Split data into 50% train and 50% test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0.7, shuffle=True)

y_train['Y'] = y_train['Y'].astype(int)
y_test['Y'] = y_test['Y'].astype(int)

from sklearn import tree
from sklearn.tree import export_text
import utils.processing as proc
import utils.visuals as visu

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)  # x_test_data

confusion_matrix = proc.get_confusion_matrix(y_test, y_predict, folders)
visu.save_confusion_matrix(confusion_matrix)


