print("Importing libraries")
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from time import perf_counter

mndata = MNIST("./MNIST/")
print("Loading testing dataset")
images_test, labels_test = mndata.load_testing()

print("Unpickling the model")
knc_pickle = open("knc.pickle", "rb")
model = pickle.load(knc_pickle)

time_testing = perf_counter()

print("Compute predictions")
predicted = model.predict(images_test)
expected = labels_test.tolist()

print("Accuracy = ", accuracy_score(expected, predicted))

time_testing = perf_counter() - time_testing
print("Time tesing = ", time_testing)
