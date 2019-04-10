print("Importing libraries")
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from skimage import io
import pickle
import numpy as np

print("Unpickling the model")
knc_pickle = open("knc.pickle", "rb")
model = pickle.load(knc_pickle)

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
