import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784',version = 1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)

def get_pred(img):
    img_pil = Image.open(img)
    imgBW = img_pil.convert('L')
    
    img_resize = imgBW.resize((28,28),Image.ANTIALIAS)
    pixelFilter = 20
    
    minPixel = np.percentile(img_resize,pixelFilter)
    img_resize_scaled=np.clip(img_resize-minPixel,0,255)
    maxPixel = np.max(img_resize)
    img_resize_scaled = np.asarray(img_resize_scaled)/maxPixel

    test_sample = np.array(img_resize_scaled).reshape(1,784)
    test_predict = lr.predict(test_sample)

    return test_predict[0]


