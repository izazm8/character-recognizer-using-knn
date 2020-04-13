import os
import cv2
import math
import numpy as np

from oct2py import Oct2Py

from skimage.feature import hog


def pad_image(image, shape):
    new_image = np.zeros(shape)
    old_row, old_col = image.shape
    new_row, new_col = shape

    for i, x in zip(range(int((new_row - old_row) / 2), int(old_row + (new_row - old_row) / 2)), range(0, old_row)):
        for j, y in zip(range(int((new_col - old_col) / 2), int(old_col + (new_col - old_col) / 2)), range(0, old_col)):
            new_image[i][j] = image[x][y]

    return new_image


def euclidean_distance(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        distance += (float(row1[i]) - float(row2[i])) ** 2  # sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 +....... )
    return math.sqrt(distance)


def read_pics(total_indices, train_indices, labels):
    features_list = []

    max_row = -1
    max_col = -1

    path = "/home/izazm8/Downloads/English/Fnt/"
    iterator = 0
    filenames = []
    for filename in os.listdir(path):
        filenames.append(filename)
    filenames.sort()

    for filename in filenames:
        for filename_ in os.listdir(path + filename):
            img = cv2.imread(path + filename + "/" + filename_, 0)
            if max_row < img.shape[0]:
                max_row = img.shape[0]
            if max_col < img.shape[1]:
                max_col = img.shape[1]

    for train_ in train_indices:
        print(train_[0])
        for train in train_:
            # print(train)
            if int(train) == 0:
                continue

            img = cv2.imread("/home/izazm8/Downloads/English/Fnt/" + total_indices[int(train) - 1] + ".png", 0)
            # print(img.shape)
            img = np.resize(img, (max_row, max_col))
            # img = pad_image(img, (max_row, max_col))
            df = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
            features = np.array(df, 'float64')

            str = np.array2string(total_indices[int(train) - 1])
            label = int(str.split('/')[2][-3:])

            features = np.append(features, [labels[label - 1]])
            features_list.append(features)

    return features_list, max_row, max_col


##########################################################################################################################


with open('/home/izazm8/Desktop/indices.txt', 'r') as file:
    data = file.read()

oc = Oct2Py()
total_indices = oc.eval(data)

with open("/home/izazm8/Desktop/train_indices.txt", "r") as file:
    data = file.read()
train_indices = oc.eval(data)

with open("/home/izazm8/Desktop/test_indices.txt", "r") as file:
    data = file.read()
test_indices = oc.eval(data)

labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
          82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
          112, 113, 114, 115, 116, 117, 118,
          119, 120, 121, 122]

print("reading pics")
features_list, max_row, max_col = read_pics(total_indices, train_indices, labels)

# test sample
TP = 0
TF = 0

for test_ in test_indices:
    for test in test_:

        if int(test) == 0:
            break

        test_row = cv2.imread("/home/izazm8/Downloads/English/Img/" + total_indices[int(test) - 1] + ".png", 0)
        # test_row = pad_image(test_row, (max_row, max_col))
        test_row = np.resize(test_row, (max_row, max_col))

        df = hog(test_row, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        test_row = np.array(df, 'float64')
        # test_row = cv2.resize(test_row, (96, 103)).flatten()

        str_ = np.array2string(total_indices[int(test) - 1])
        label = int(str_.split('/')[2][-3:])

        # getting neighbours of test sample
        print("Calculating distance..")
        distances = []
        for train_row in features_list:
            distances.append((train_row, euclidean_distance(train_row, test_row)))

        print(len(distances))
        print("sorting..")
        distances.sort(key=lambda tup: tup[1])

        # calculating neighbours
        print("calculating neighbours..")
        neighbour = 5
        neighbours = []
        for i in range(neighbour):
            neighbours.append(distances[i][0])
        for neg in neighbours:
            print(neg)

        # getting classes of neighbours
        print("getting classes and predicting..")
        neighbour_classes = [row[-1] for row in neighbours]
        for class_ in neighbour_classes:
            print("Neg: " + str(class_))
        prediction = max(set(neighbour_classes), key=neighbour_classes.count)
        type(prediction)
        if int(prediction) == labels[label - 1]:
            TP = TP + 1
        else:
            TF = TF + 1

        print("Predicted Digit: " + str(chr(int(prediction))))  # + "      " + str(labels[label-1]))
        break

accuracy = (TP / (TP + TF)) * 100
print("acuuracy: " + accuracy)
