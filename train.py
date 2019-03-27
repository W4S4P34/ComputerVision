print('Importing struct')
import struct
print('Importing numpy')
import numpy as np
print('Importing sklearn.neighbors + sklearn.metrics')
from sklearn import neighbors, metrics
print('Importing skimage.io')
from skimage import io
print('Importing itertools')
import itertools
print('Importing pickle')
import pickle

print('Reading MNIST training image set')
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB',f.read(4))
        shape = tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
        # return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        # Supress deprecation warning: fromstring() --> frombuffer()
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
raw_train = read_idx("MNIST/train-images.idx3-ubyte")
train_data = np.reshape(raw_train,(60000,28*28))
train_label = read_idx("MNIST/train-labels.idx1-ubyte")

print('Classifying all images')
#idx = (train_label == 0) | (train_label == 1) | (train_label == 2) | (train_label == 3) | (train_label == 4) | (train_label == 5) | (train_label == 6) | (train_label == 7) | (train_label == 8) | (train_label == 9)
#X = train_data[idx]
#Y = train_label[idx]
X = train_data
Y = train_label
knn = neighbors.KNeighborsClassifier(n_neighbors = 5, weights='distance', algorithm='ball_tree').fit(X,Y)

print('Pickling the model')
pfile = open("knn.pickle", 'wb')
pickle.dump(knn, pfile)

print('All done.')
