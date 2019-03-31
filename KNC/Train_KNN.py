print("Importing libraries")
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
import pickle
from time import perf_counter

mndata = MNIST("./MNIST/")
print("Loading training dataset")
images_train, labels_train = mndata.load_training()

# print("Limiting training dataset")
# images_train = images_train[:10000]
# labels_train = labels_train[:10000]

print("Configuring the model")
model = KNeighborsClassifier(n_neighbors = 10,
                             weights = 'distance',
                             algorithm = 'ball_tree',
                             leaf_size = 50,
                             # p = 2,
                             metric = 'euclidean',
                             n_jobs = -1
                            )

training_time = perf_counter()

print("Building the model")
model.fit(images_train, labels_train)

print("Pickling the model")
knc_pickle = open("knc.pickle", "wb")
pickle.dump(model, knc_pickle)

print("All done.")
training_time = perf_counter() - training_time
print("Training time = ", training_time)
