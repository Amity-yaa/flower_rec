import numpy as np
np.random.seed(42)
import tensorflow.compat.v1 as tf
import keras as K
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import backend
# from numba import cuda
from keras.models import load_model
import time 
from sklearn.svm import SVC

# import tensorflow as tf # tf.__version__ == 2.1.0, keras.__version__ == 2.3.1
# https://blog.csdn.net/u012388993/article/details/102573008
# import keras.backend.tensorflow_backend as ktf
# https://blog.csdn.net/zuoyouzouzou/article/details/104329286
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as ktf

# GPU 显存自动调用
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)

train_path = r'flowers_google'
labels_path = "flowers_label.csv"
ids_path = "flowers_idx.csv"


def PrepareAnnotation(ids_path, labels_path, train_path):
    ids = pd.read_csv(ids_path)
    labels = pd.read_csv(labels_path)

    annotation_dict = {}

    for i in range(len(ids['id'])):
        image_id = ids['id'][i]
        image_lable = ids['flower_cls'][i]
        annotation_dict[image_id] = image_lable

    label_dict = {}
    num_classes = 0

    for i in range(len(labels['flower_class'])):
        label_dict[labels['flower_class'][i]] = labels['label'][i]
        num_classes += 1

    for key, value in annotation_dict.items():
        annotation_dict[key] = label_dict[value]

    image_pathes = []
    targets = []

    for key, value in annotation_dict.items():
        image_name = str(key) + ".jpeg"
        image_path = os.path.join(train_path, image_name)
        image_pathes.append(image_path)
        targets.append(value)

    return image_pathes, targets, num_classes

image_pathes, targets, num_classes = PrepareAnnotation(ids_path, labels_path, train_path)
val_ratio = 0.2
x_train, x_val, y_train, y_val = train_test_split(image_pathes, targets, test_size=val_ratio, random_state=42, shuffle = True, stratify = targets)
print("num_train = ", len(x_train))
print("num_valid = ", len(x_val))

batch_size = 32 # Tesla 上占用9013MB
best_model_root = 'models/models_'+str(batch_size)
model_root = 'models/model_ann'

model_pretrained = load_model(os.path.join(best_model_root, 'Best_Model.h5'))
for layer in model_pretrained.layers:
    if isinstance(layer, K.layers.GlobalAveragePooling2D):
        gap_layer =  layer
model_pretrained = K.models.Model(model_pretrained.inputs, gap_layer.output)

for layer in model_pretrained.layers:
    if isinstance(layer, K.layers.Dense):
        layer.trainable = True
    else:
        layer.trainable = False


def files2features(x_train):
    x_svm = []
    from tqdm import tqdm
    batch_num = int(np.ceil(len(x_train) / batch_size))
    bar = tqdm(total=batch_num)
    for i in range(batch_num):
        begin_i = i * batch_size
        end_i = (i + 1) * batch_size
        x = x_train[begin_i:end_i]
        x_batch = []

        for i in range(len(x)):
            image = cv2.imread(x[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = K.applications.resnet50.preprocess_input(image)
            x_batch.append(image)

        x_batch = np.array(x_batch)
        x_batch = model_pretrained.predict(x_batch).reshape(x_batch.shape[0], -1)
        x_svm.append(x_batch)

        bar.update(1)
    x_svm = np.concatenate(x_svm, axis=0)
    return x_svm

feature_train = files2features(x_train)
y_train = np.array(y_train).ravel()
feature_val = files2features(x_val)
y_val = np.array(y_val).ravel()

save_root = 'data_for_svm'
if not os.path.exists(save_root):
    os.mkdir(save_root)
np.save(os.path.join(save_root, 'x_train'), feature_train)
np.save(os.path.join(save_root, 'y_train'), y_train)
np.save(os.path.join(save_root, 'x_val'), feature_val)
np.save(os.path.join(save_root, 'y_val'), y_val)


from sklearn.feature_selection import SelectFromModel