import struct
import numpy as np
from sklearn import neighbors,metrics
import matplotlib.pyplot as plt
import itertools

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

#train data 2-3-8

idx = (train_label == 0) | (train_label == 1) | (train_label == 2) | (train_label == 3) | (train_label == 4) | (train_label == 5) | (train_label == 6) | (train_label == 7) | (train_label == 8) | (train_label == 9)
X = train_data[idx]
Y = train_label[idx]
knn = neighbors.KNeighborsClassifier(n_neighbors = 5).fit(X,Y)

#test data 2-3-8

idx = (test_label == 0) | (test_label == 1) | (test_label == 2) | (test_label == 3) | (test_label == 4) | (test_label == 5) | (test_label == 6) | (test_label == 7) | (test_label == 8) | (test_label == 9)
x_test = test_data[idx]
y_true = test_label[idx]
y_predict = knn.predict(x_test)


#confusion matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.imshow(cm,interpolation = "nearest" , cmap =cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 45)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j],fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = metrics.confusion_matrix(y_true, y_predict)
plot_confusion_matrix(cm,
                      ['0','1','2','3','4','5','6','7','8','9'],
                      True)