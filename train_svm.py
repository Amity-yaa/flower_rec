import pandas as pd
import keras as K
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras import models, layers
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image
import os
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
import joblib
import numpy as np
import os

feature_root = 'data_for_svm'
x_train = np.load(os.path.join(feature_root, 'x_train.npy'))
y_train = np.load(os.path.join(feature_root, 'y_train.npy'))
x_test = np.load(os.path.join(feature_root, 'x_val.npy'))
y_test = np.load(os.path.join(feature_root, 'y_val.npy'))

svm = SVC(kernel='linear').fit(x_train, y_train)

score_train = svm.score(x_train, y_train)
score_test = svm.score(x_test, y_test)
print('training acc,', score_train)
print('testing acc,', score_test)
joblib.dump(svm, 'models/svm/2021-10-25_V100_{}.svm'.format(score_test))
