print("Importing python-mnist")
from mnist import MNIST
print("Importing sklearn.svm.SVC")
from sklearn.svm import SVC
print('Importing skimage.io')
from skimage import io
print("Importing pickle")
import pickle
print("Importing numpy")
import numpy as np

print("Unpickling the model")
svc_pickle = open("svc.pickle", "rb")
model = pickle.load(svc_pickle)

test_file = 'Test.bmp'

print('Reading', test_file)
test_data = io.imread(test_file, as_gray=True)
print('Normalizing image and converting into ndarray')
test_data *= 255
test_data -= 255
test_data = np.abs(test_data)
test_data = test_data.astype(int)
test_data = np.reshape(test_data,(1,28*28))
print(test_data)

print("Predicting test image")
predicted = model.predict(test_data)

print(predicted)
