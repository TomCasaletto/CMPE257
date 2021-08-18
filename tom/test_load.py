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

# Let's get a subset of the data for testing
num_train=1000
num_test=100
X_train = np.empty((num_train,3,4096),dtype='float64')
y_train = np.empty((num_train),dtype='float64')
for i in range(num_train):
    X_train[i,:,:]=np.load(train.iloc[i][2])
    y_train[i] = train.iloc[i][1]
print(X_train.shape)
print(X_train)

X_test = np.empty((num_test,3,4096),dtype='float64')
y_test = np.empty((num_test),dtype='float64')
for i in range(num_test):
    X_test[i,:,:]=np.load(test.iloc[i][2])
    y_test[i] = test.iloc[i][1]
print(X_test.shape)
print(X_test)

#with open('X_train.npy', 'wb') as f:
#    np.save(f, X_train, allow_pickle=False)
#with open('y_train.npy', 'wb') as f:
#    np.save(f, y_train, allow_pickle=False)
#with open('X_test.npy', 'wb') as f:
#    np.save(f, X_test, allow_pickle=False)
#with open('y_test.npy', 'wb') as f:
#    np.save(f, y_test, allow_pickle=False)

#X_train = X_train.reshape(X_train.shape[0], -1)
#np.savetxt('X_train.csv', X_train, delimiter=',')
#np.savetxt('y_train.csv', y_train, delimiter=',')
#X_test = X_test.reshape(X_test.shape[0], -1)
#np.savetxt('X_test.csv', X_test, delimiter=',')
#np.savetxt('y_test.csv', y_test, delimiter=',')

np.savez('gwz', X_train, y_train, X_test, y_test)

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
