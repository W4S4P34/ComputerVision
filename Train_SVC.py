print("Importing python-mnist")
from mnist import MNIST
print("Importing sklearn.svm.SVC")
from sklearn.svm import SVC
print("Importing pickle")
import pickle
print("Importing time.perf_counter")
from time import perf_counter

mndata = MNIST("./MNIST/")
print("Loading training dataset")
images_train, labels_train = mndata.load_training()

print("Limiting training dataset")
images_train = images_train[:1000]
labels_train = labels_train[:1000]

print("Configuring SVC")
model = SVC(C = 5.0,
            kernel = 'rbf',
            gamma = 0.01,          # auto, scale
            shrinking = True,
            probability = False,                # probability estimates = slower fitting
            max_iter = -1,                      # -1 = infinity
            cache_size = 1000,                  # cache size in MB
            decision_function_shape = 'ovr',    # ovo/ovr
            verbose = True
           )

training_time = perf_counter()

print("Building the model")
model.fit(images_train, labels_train)

print("Pickling the model")
svc_pickle = open("svc.pickle", "wb")
pickle.dump(model, svc_pickle)

print("All done.")
training_time = perf_counter() - training_time
print("Training time = ", training_time)
