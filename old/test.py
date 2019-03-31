print('Importing numpy')
import numpy as np
print('Importing sklearn.neighbors + sklearn.metrics')
from sklearn import neighbors, metrics
print('Importing skimage.io')
from skimage import io
print('Importing pickle')
import pickle

print('Unpickling the model')
pfile = open('knn.pickle', 'rb')
knn = pickle.load(pfile)

# print('Reading MNIST test data')
# def read_idx(filename):
#     with open(filename, 'rb') as f:
#         zero, data_type, dims = struct.unpack('>HBB',f.read(4))
#         shape = tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
#         # return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
#         # Supress deprecation warning: fromstring() --> frombuffer()
#         return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
# raw_test = read_idx("C:/Users/DELL-7559/Downloads/t10k-images.idx3-ubyte")
# test_data = np.reshape(raw_test,(10000,28*28))
# test_label = read_idx("C:/Users/DELL-7559/Downloads/t10k-labels.idx1-ubyte")

test_file = 'test.bmp'

print('Reading', test_file)
test_data = io.imread(test_file, as_gray=True)
print('Normalizing image and converting into ndarray')
test_data *= 255
test_data -= 255
test_data = np.abs(test_data)
test_data = test_data.astype(int)
test_data = np.reshape(test_data,(1,28*28))
# print(test_data)

print('Predicting data')
# idx = (test_label == 0) | (test_label == 1) | (test_label == 2) | (test_label == 3) | (test_label == 4) | (test_label == 5) | (test_label == 6) | (test_label == 7) | (test_label == 8) | (test_label == 9)
# x_test = test_data[idx]
# y_true = test_label[idx]
# y_predict = knn.predict(test_data)
# print(y_predict)
y_predict = knn.predict(test_data)
print(y_predict)

# print('Importing matplotlib.pyplot')
# import matplotlib.pyplot as plt
# 
# print('Drawing confusion matrix')
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title="Confusion matrix",
#                           cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
# 
#     print(cm)
#     
#     plt.imshow(cm,interpolation = "nearest" , cmap =cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks,classes,rotation = 45)
#     plt.yticks(tick_marks,classes)
# 
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
#         plt.text(j, i, format(cm[i,j],fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#     
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     
# cm = metrics.confusion_matrix(y_true, y_predict)
# plot_confusion_matrix(cm,
#                       ['0','1','2','3','4','5','6','7','8','9'],
#                       True)
