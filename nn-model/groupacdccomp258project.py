# -*- coding: utf-8 -*-
"""GroupACDCCOMP258Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14nmbX2DyiCHkplTA_MXVgTYN0jdKU5Xm

# **A. Import Libraries**
"""

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Normalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from scipy import stats
import pandas as pd
import numpy as np

import pathlib
import pprint
import tempfile

"""# **B. Data Processing**"""

colum_names = pd.read_csv('https://raw.githubusercontent.com/annyper/Bicycle_Thefts/main/Student%20data.csv', nrows=18).transpose().head(1)
colum_names

colum_names=colum_names.drop(columns=[0,15,16])
colum_names=colum_names.squeeze()
colum_names=colum_names.str.replace('numeric', '', regex=True)
colum_names=colum_names.str.replace('\'', '', regex=True)
colum_names=colum_names.str.replace(' ', '', regex=True)
colum_names=colum_names.str.replace('\{1,0\}', '', regex=True)
colum_names = list(colum_names)
colum_names

# Class Names
class_names = ['FirstTermGpa', 'SecondTermGpa', 'FirstLanguage', 'Funding', 'School', 'FastTrack', 'Coop', 'Residency', 'Gender', 'PreviousEducation', 'AgeGroup', 'HighSchoolAverageMark', 'MathScore', 'EnglishGrade', 'FirstYearPersistence']
class_names_categorical = ['FirstLanguage', 'Funding', 'School', 'FastTrack', 'Coop', 'Residency', 'Gender', 'PreviousEducation', 'AgeGroup']
class_names_numerical = ['FirstTermGpa', 'SecondTermGpa', 'HighSchoolAverageMark', 'MathScore', 'EnglishGrade']
label_dict = {'FirstYearPersistence_no': 0, 'FirstYearPersistence_yes': 1}

students_ds = pd.read_csv('https://raw.githubusercontent.com/annyper/Bicycle_Thefts/main/Student%20data.csv',skiprows=24, names=colum_names)
students_ds = students_ds.replace('?', np.nan)
students_ds

students_ds=students_ds.astype('float64')

scaler = StandardScaler()
scaler.fit(students_ds)
ds_mean = scaler.mean_
ds_mode = np.array(students_ds.mode())[0]
ds_std = scaler.scale_
students_ds.dtypes

# Define feature columns
feature_columns = [
    tf.feature_column.numeric_column("FirstTermGpa"),
    tf.feature_column.numeric_column("SecondTermGpa"),
    tf.feature_column.categorical_column_with_identity("FirstLanguage", num_buckets=3),
    tf.feature_column.categorical_column_with_identity("Funding", num_buckets=9),
    tf.feature_column.categorical_column_with_identity("School", num_buckets=6),
    tf.feature_column.categorical_column_with_identity("FastTrack", num_buckets=2),
    tf.feature_column.categorical_column_with_identity("Coop", num_buckets=2),
    tf.feature_column.categorical_column_with_identity("Residency", num_buckets=2),
    tf.feature_column.categorical_column_with_identity("Gender", num_buckets=2),
    tf.feature_column.categorical_column_with_identity("PreviousEducation", num_buckets=3),
    tf.feature_column.categorical_column_with_identity("AgeGroup", num_buckets=8),
    tf.feature_column.numeric_column("HighSchoolAverageMark"),
    tf.feature_column.numeric_column("MathScore"),
    tf.feature_column.numeric_column("EnglishGrade")
]

tensor = tf.convert_to_tensor(students_ds.values)
ds = tf.data.Dataset.from_tensor_slices(dict(students_ds))
ds

np.random.seed(42)
tf.random.set_seed(42)

# Define a function to replace NaN values with the column mean
def preprocess(features):
    for feature_name, feature_value in features.items():
        # Replace NaN values with mean using TensorFlow API
        if feature_name in class_names_numerical:
                mean = ds_mean[colum_names.index(feature_name)] 
                std = ds_std[colum_names.index(feature_name)]
                features[feature_name] = tf.where(tf.math.is_nan(feature_value),mean, feature_value)
                #features[feature_name] = (features[feature_name] - mean) / std

        # Replace NaN values with mode using TensorFlow API
        if feature_name in class_names_categorical:
                mode = ds_mode[colum_names.index(feature_name)]
                features[feature_name] = tf.where(tf.math.is_nan(feature_value),mode, feature_value)

    return features

# Apply the transformation function to the dataset
ds = ds.map(preprocess)

ds_mode

ds_std

ds_mean

# create a batch of data
batch = next(iter(ds.batch(len(ds))))
# create a pandas dataframe from the numpy array
df = pd.DataFrame(batch, columns=colum_names)
df

df['HighSchoolAverageMark'].unique()

df['FirstLanguage'].unique()

"""*Classify the data into X, and Y (Features and Targets)*"""

# Extract target values according to statement
y = pd.get_dummies(df.iloc[0:, 14])
y = np.array(y)

# Extract features
X = df.iloc[0:, 0:14].values

X, y

"""# **C. SPLITTING DATA**

#### Splitting the data
"""

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = 0.1, shuffle = True)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size = 0.2, shuffle = True)

X_train_full.shape, y_train_full.shape

X_train.shape, y_train.shape

X_test.shape, y_test.shape

X_val.shape, y_val.shape

n_features = X_train.shape[1]
print(n_features)

"""# **D. TRAINING MODEL**"""

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()

# input layer
model.add(Dense(n_features))

model.add(Normalization(axis=None))

# add the first hidden layer with 300 neurons, relu  activation function
model.add(Dense(300, activation="relu", kernel_initializer='he_normal'))

# add the first hidden layer with 150 neurons, relu  activation function
model.add(Dense(150, activation="relu", kernel_initializer='he_normal'))

model.add(keras.layers.AlphaDropout(rate=0.2))

# add the second hidden layer with 80 neurons, relu  activation function
model.add(Dense(80, activation="relu", kernel_initializer='he_normal'))

# add the second hidden layer with 25 neurons, relu  activation function
model.add(Dense(25, activation="relu", kernel_initializer='he_normal'))

# add the output layer with 2 neurons, erlu activation function
model.add(Dense(2, activation="softmax"))

#optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
optimizer = tf.keras.optimizers.Nadam(lr=5e-4)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)

checkpoint = ModelCheckpoint('save_model_tf/best-model-{epoch:03d}.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history = model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data = (X_val, y_val), callbacks=[checkpoint, early_stopping])

model.summary()

"""# **E. EVALUATION MODEL**"""

# Evaluate with training data
loss, acc = model.evaluate(X_train, y_train, verbose=0)
print('Test Accuracy: %.3f' % acc)

# Evaluate with testing data
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# Evaluate with val data
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print('Test Accuracy: %.3f' % acc)

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 139), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 139), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 139), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 139), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# make predictions on the testing set
predict = model.predict(X_test, batch_size=32)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predict_labeled = np.argmax(predict, axis=1)

# show a nicely formatted classification report
print(classification_report(y_test.argmax(axis=1), predict_labeled, target_names=label_dict))

'''
0	0	1	2	6	2	1	1	2	1	1	59	16	7	1
2.5	2	3	4	6	1	2	2	2	1	3	NaN	NaN	7	1
4.25	3.923077	1	1	6	2	1	1	1	2	3	92	41	9	1
'''

X_test[0]

row = [0, 0, 1, 2, 6, 2, 1, 1, 2, 1, 1, 59, 16, 7]
y_hat = model.predict([row])
print('Predicted: %s (class=%d)' % (y_hat, y_hat.argmax(axis=1)))

row = [2.5, 2, 3, 4, 6, 1, 2, 2, 2, 1, 3, 80, 80, 7]
y_hat = model.predict([row])
print('Predicted: %s (class=%d)' % (y_hat, y_hat.argmax(axis=1)))

row = [4, 3, 1, 1, 6, 2, 1, 1, 1, 2, 3, 92, 41, 9]
y_hat = model.predict([row])
print('Predicted: %s (class=%d)' % (y_hat, y_hat.argmax(axis=1)))

row = [1, 1, 3, 4, 6, 2, 2, 2, 2, 2, 0, 92, 41, 9]
y_hat = model.predict([row])
print('Predicted: %s (class=%d)' % (y_hat, y_hat.argmax(axis=1)))

"""# **F. SAVE MODEL**"""

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_last_model.h5')

# Save Data
print(X_train, type(X_train))

# Convert the NumPy array to a Pandas DataFrame
X_train_df = pd.DataFrame(X_train)
X_val_df = pd.DataFrame(X_val)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train)
y_val_df = pd.DataFrame(y_val)
y_test_df = pd.DataFrame(y_test)

# Write the data to a CSV file
X_train_df.to_csv('X_train_data_group_acdc.csv',index=False)
X_val_df.to_csv('X_val_data_group_acdc.csv',index=False)
X_test_df.to_csv('X_test_data_group_acdc.csv',index=False)

y_train_df.to_csv('y_train_data_group_acdc.csv',index=False)
y_val_df.to_csv('y_val_data_group_acdc.csv',index=False)
y_test_df.to_csv('y_test_data_group_acdc.csv',index=False)

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_last_model.h5')

# Show the model architecture
new_model.summary()

row = [0, 0, 1, 2, 6, 2, 1, 1, 2, 1, 1, 59, 16, 7]
y_hat = new_model.predict([row])
print('Predicted: %s (class=%d)' % (y_hat, y_hat.argmax(axis=1)))

row = [4, 3, 1, 1, 6, 2, 1, 1, 1, 2, 3, 92, 41, 9]
y_hat = new_model.predict([row])
print('Predicted: %s (class=%d)' % (y_hat, y_hat.argmax(axis=1)))