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
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
]
y_out = pd.DataFrame(columns=list('Y'))
x_out = pd.DataFrame(columns=range(128*128))
for folder in folders:
    outY, outX = load_images_from_folder("C:/data/PlantVillage-Dataset/raw/grayscale/", folder, folders)
    y_out = pd.concat([y_out, outY])
    x_out = pd.concat([x_out, outX])
y_out = y_out.reset_index(drop=True)
x_out = x_out.reset_index(drop=True)
print(y_out.tail())

    # your code that does something with the return images goes here
