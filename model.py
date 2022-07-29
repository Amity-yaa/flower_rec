import keras as K
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend
# from numba import cuda
from keras.models import load_model


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
x_train, x_val, y_train, y_val = train_test_split(image_pathes, targets, test_size=val_ratio, random_state=42, shuffle = True)
print("num_train = ", len(x_train))
print("num_valid = ", len(x_val))


def Generator(X, Y, batch_size):
    while True:
        indexes = np.random.choice(len(X), batch_size)
        x = []
        y = []

        for index in indexes:
            x.append(X[index])
            y.append(Y[index])

        x_batch = []

        y_batch = K.utils.to_categorical(y, num_classes=num_classes)

        for i in range(len(x)):
            image = cv2.imread(x[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = K.applications.resnet50.preprocess_input(image)
            x_batch.append(image)

        x_batch = np.array(x_batch)

        yield x_batch, y_batch

def ResNet50Model():
    model = K.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    new_output = K.layers.GlobalAveragePooling2D()(model.output)
    new_output = K.layers.Dense(num_classes, activation='softmax')(new_output)

    model = K.engine.training.Model(model.inputs, new_output)

    return model

model = ResNet50Model()

# Freez first 168 Layers Except the Batch Normalization ones.
num_feerezed_layer = 168

for layer in model.layers:
    layer.trainable = True

    if isinstance(layer, K.layers.BatchNormalization):
        # Set Batch Norm momentun to 0.9 to faster adapt to the new DataSet.
        layer.momentum = 0.9

for layer in model.layers[:num_feerezed_layer]:

    if not isinstance(layer, K.layers.BatchNormalization):
        layer.trainable = False

# Check if the trainable layers are set Correctly
for layer in model.layers:
    print(layer.name, " ", "trainable = ", layer.trainable)

checkpoint = ModelCheckpoint('Best_Model.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')
model.compile(loss='categorical_crossentropy', optimizer = K.optimizers.Adamax(lr=0.01), metrics=['accuracy'])
batch_size = 32
model.fit(Generator(x_train, y_train, batch_size),
          steps_per_epoch=len(x_train)//batch_size,
          epochs= 4,
          validation_data=Generator(x_val, y_val, batch_size),
          validation_steps=len(x_val) // batch_size,
          callbacks=[checkpoint])