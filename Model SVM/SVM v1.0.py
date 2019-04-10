#Get images and Get labels

def get_images(img_file, number):
    f = open(img_file, "rb") # Open file in binary mode
    f.read(16) # Skip 16 bytes header
    images = []

    for i in range(number):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    return images

def get_labels(label_file, number):
    l = open(label_file, "rb") # Open file in binary mode
    l.read(8) # Skip 8 bytes header
    labels = []
    for i in range(number):
        labels.append(ord(l.read(1)))
    return labels

#Output image MNIST

import os
import numpy as np
from skimage import io
from skimage.feature import hog

def convert_png(images, labels, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    for i in range(len(images)):
        out = os.path.join(directory, "%06d-num%d.png"%(i,labels[i]))
        io.imsave(out, np.array(images[i]).reshape(28,28))
    
number = 100
train_images = get_images("D:/GITHUB/Computer-Vision-12/MNIST/train-images-idx3-ubyte", number)
train_labels = get_labels("D:/GITHUB/Computer-Vision-12/MNIST/train-labels-idx1-ubyte", number)

# =============================================================================
# convert_png(train_images, train_labels, "D:/GITHUB/Computer-Vision-12/MNIST/MNIST/Images")
# =============================================================================

#Convert binary file (MNIST) to csv

def output_csv(images, labels, out_file):
    o = open(out_file, "w")
    for i in range(len(images)):
        o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
    o.close()
    




#Normalize dataset

train_images = np.array(train_images)/255

#Train data
from sklearn import svm, metrics

print("TRAIN")
TRAINING_SIZE = 60000
train_images = get_images("D:/GITHUB/Computer-Vision-12/MNIST/train-images-idx3-ubyte", TRAINING_SIZE)
train_images = np.array(train_images)/255
train_labels = get_labels("D:/GITHUB/Computer-Vision-12/MNIST/train-labels-idx1-ubyte", TRAINING_SIZE)

#HOG
# =============================================================================
# X1 = []
# 
# for pixels in train_images:
# 	image = []
# 	for i in range(0, 28):
# 		image.append(pixels[i * 28 : (i + 1) * 28])
# 	np.multiply(image, 255)	
# 	fd = hog(image, orientations=12, pixels_per_cell=(4, 4), 
# 					cells_per_block=(3, 3), visualise=False)
# 	X1.append(fd)
# 
# =============================================================================

#MODELING

clf = svm.SVC(C=5, gamma = 0.00001)
clf.fit(train_images, train_labels)

TEST_SIZE = 500
test_images = get_images("D:/GITHUB/Computer-Vision-12/MNIST/t10k-images-idx3-ubyte", TEST_SIZE)
test_images = np.array(test_images)/255
test_labels = get_labels("D:/GITHUB/Computer-Vision-12/MNIST/t10k-labels-idx1-ubyte", TEST_SIZE)

print("PREDICT")
predict = clf.predict(test_images)

print("RESULT")
ac_score = metrics.accuracy_score(test_labels, predict)
cl_report = metrics.classification_report(test_labels, predict)
print("Score = ", ac_score)
print(cl_report)

from joblib import dump
dump(clf, 'mnist-svm.joblib') 