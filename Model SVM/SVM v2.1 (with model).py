from joblib import load
from skimage import io
import numpy as np

clf = load('mnist-svm.joblib') 

test_file = 'C:/Users/DELL-7559/Desktop/test.png'
test_data = io.imread(test_file, as_gray=True)

#Predict
print(test_data)
test_data = np.reshape(test_data,(1,28*28))

predicted = clf.predict(test_data)
print(predicted)