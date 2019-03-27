import struct
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import operator
#import scipy
from scipy.special import expit

#reading dataset func
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB',f.read(4))
        shape = tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

raw_train = read_idx("C:/Users/DELL-7559/Downloads/train-images.idx3-ubyte")
train_data = np.reshape(raw_train,(60000,28*28))
train_label = read_idx("C:/Users/DELL-7559/Downloads/train-labels.idx1-ubyte")

raw_test = read_idx("C:/Users/DELL-7559/Downloads/t10k-images.idx3-ubyte")
test_data = np.reshape(raw_test,(10000,28*28))
test_label = read_idx("C:/Users/DELL-7559/Downloads/t10k-labels.idx1-ubyte")

#KNN func

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))

def absolute_distance(vector1, vector2):
    return np.sum(np.absolute(vector1-vector2))

def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbors = []
    for i in range (0, X_train.shape[0]):
        dist = absolute_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range (k):
        #print distances[x]
        neighbors.append(distances[x][0])
    return neighbors

def predictkNNClass(output, y_train):
    classVotes = {}
    for i in range(len(output)):
#         print output[i], y_train[output[i]]
        if y_train[output[i]] in classVotes:
            classVotes[y_train[output[i]]] += 1
        else:
            classVotes[y_train[output[i]]] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print sortedVotes
    return sortedVotes[0][0]

def kNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    for i in range(0, X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes


#train data 

instance_neighbours = get_neighbours(train_data, test_data, 9)
indices = instance_neighbours
for i in range(9):
    plt.subplot(3,3,i + 1)
    plt.imshow(train_data[indices[i]].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Index {} Class {}".format(indices[i], train_label[indices[i]]))
    plt.tight_layout()

