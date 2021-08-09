import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns

# Following load code adapted from:
#  https://www.kaggle.com/xuzongniubi/g2net-efficientnet-b7-baseline-training

train = pd.read_csv('./data/training_labels.csv')
test = pd.read_csv('./data/sample_submission.csv')

def get_train_file_path(image_id):
    return "./data/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "./data/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

train['file_path'] = train['id'].apply(get_train_file_path)
test['file_path'] = test['id'].apply(get_test_file_path)

print(train.info)
print(test.info)

print(train.head())
print(test.head())

# Let's plot the time series from the first training example
print('First example')
print(train.iloc[0][2])
data = np.load(train.iloc[0][2])

print(type(data))
print(data.shape)
print(data.dtype)
print(data.strides)
print(data.data)
print(data)

plt.figure(1)
plt.clf()
idx = range(0, 4096)
#for series in range(3):
#    plt.plot(idx, data[series][:])
plt.plot(idx, data[0][:], label='Hanford')
plt.plot(idx, data[1][:], label='Livingston')
plt.plot(idx, data[2][:], label='Virgo')

plt.xlabel('time (idx)')
plt.ylabel('signal')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.title('File: ' + train.iloc[0][2] + ', target=' + str(train.iloc[0][1]))


print('Second example')
print(train.iloc[1][2])
data = np.load(train.iloc[1][2])

plt.figure(2)
plt.clf()
idx = range(0, 4096)
#for series in range(3):
#    plt.plot(idx, data[series][:])
plt.plot(idx, data[0][:], label='Hanford')
plt.plot(idx, data[1][:], label='Livingston')
plt.plot(idx, data[2][:], label='Virgo')

plt.xlabel('time (idx)')
plt.ylabel('signal')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.title('File: ' + train.iloc[1][2] + ', target=' + str(train.iloc[1][1]))

plt.show()
