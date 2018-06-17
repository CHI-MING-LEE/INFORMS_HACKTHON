import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import glob
from pathlib import PurePath
import random

# from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
os.getcwd()
os.chdir('E:\\Users\\Ross\\Downloads\\Python\\INFORMS競賽\\informs hackathon\\ISIC-2017_Training_Data')
ORIGIN_IMG_PATH = 'E:\\Users\\Ross\\Downloads\\Python\\INFORMS競賽\\informs hackathon\\ISIC-2017_Training_Data'

all_img_paths = glob.glob(os.path.join(ORIGIN_IMG_PATH, "*.jpg"))
# for i in os.listdir('/Users/championlin/Desktop/ISIC-2017_Training_Data'):
#     if 'superpixels' in i:
#         os.remove('/Users/championlin/Desktop/ISIC-2017_Training_Data/{}'.format(i))
df = pd.read_csv('E:/Users/Ross/Downloads/Python/INFORMS競賽/informs hackathon/ISIC-2017_Training_Part3_GroundTruth.csv')
# 抽600個
# df['catacategory'].value_counts()
# idx = [np.where(df['catacategory'] == i) for i in range(3)]
# idx2 = [random.sample(list(idx[j][0]), 200) for j in range(3)]
# idx3 = idx2[0] + idx2[1] + idx2[2]
# df2 = df.loc[idx3]

# 將圖片載入並進行resize
img_size = 400
X = np.zeros(shape=(2000, img_size, img_size, 3), dtype=np.float32)
Y = np.zeros(shape=(2000,), dtype=np.float32)

for i, name in enumerate(df['image_id']):
    print(i)
    im = cv2.imread('{}.jpg'.format(name))
    im_resize = cv2.resize(im, (img_size, img_size))
    X[i] = im_resize
    Y[i] = df.loc[df['image_id'] == name, 'catacategory']

y = np_utils.to_categorical(Y, 3)

# import matplotlib.pyplot as plt
#
# plt.imshow(X[0])
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# 建立訓練模型 - 使用inception_resnet_v2
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2

model = InceptionResNetV2(include_top=True, weights='imagenet')
model.layers.pop() # 去掉最後一層 ( 倒數第二層就是GAP
new_layer = Dense(3, activation='softmax', name='my_dense')

inp = model.input
out = new_layer(model.layers[-1].output)

model2 = Model(inp, out)

def preprocess_input(x):
    x /= 255.
    # x -= 0.5
    # x *= 2.
    return x

x_train_p = preprocess_input(x_train)
x_test_p = preprocess_input(x_test)

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(x_train_p, y_train, batch_size=10, epochs=20, validation_data=(x_test_p, y_test))
# model2.summary()

pred = map(np.argmax, model2.predict(x_test_p))
truth = map(np.argmax, y_test)

truth = pd.DataFrame(list(truth))
pred = pd.DataFrame(list(pred))

truth[0].value_counts()
pred[0].value_counts()

from keras.models import load_model
# load聖凱model
model_lin = load_model("E:\\Users\\Ross\\Downloads\\Python\\INFORMS競賽\\informs hackathon\\model_hack.h5")

#
# Freeze trainable
# for layer in model_inception.layers[:-1]:
#     layer.trainab000000B06C07C00 of size 256

# InceptionV3
model_inception = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                               classes=1000)

model_inception.summary()

SOFT_MAX = Dense(units=3, activation='softmax')(model_inception.layers[-2].output)
model = Model(model_inception.input, SOFT_MAX)
model.summary()

for layer in model2.layers[:-1]:
    layer.trainable = False

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# path_checkpoint = 'checkpoint.keras'
# callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)
# callback_tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)
# callbacks = [callback_checkpoint, callback_tensorboard]
epochs = 10
model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                    steps_per_epoch=len(x_train) / 128, epochs=epochs)  # , callbacks=callbacks)
# model.fit(x_train, y_train, batch_size=16, epochs=epochs, validation_data=[x_test, y_test])  # , callbacks=callbacks)

# 模型預測
def model_pred(cute_model=None, img_matrix=None, W=299, H=299, ch=3):
    illness_dict = {'0': "Melanoma", "1": "Seborrheic Keratosis", "2": "Normal"}
    # img_matrix的維度要是(1, W, H, ch)
    img_matrix = img_matrix.reshape(1, W, H, ch)
    result = cute_model.predict(img_matrix)
    illness = illness_dict[str(np.argmax(result))]

    return illness