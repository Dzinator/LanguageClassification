'''
    K-Nearest Neighbors implementation
    
    TODO cross validation within training data
    
    Yanis
'''

import csv
import random


def read_data(train_data=[], test_data=[]):
    with open('id_vector_train.csv', 'rb') as csv_data:
        lines = csv.reader(csv_data)

        for l in lines:
            train_data.append(l)

    with open('id_vector_test.csv', 'rb') as csv_data:
        lines = csv.reader(csv_data)

        for l in lines:
            test_data.append(l)


def hamming_distance(str1, str2):
    # Works only on strings for now
    if len(str1) == len(str2):
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))
    else:
        return float('nan')


def euclidean_distance(row1, row2):
    if len(row1) == len(row2):
        distance = 0.0
        for x, y in zip(row1, row2):
            distance += pow((x - y), 2)

        return math.sqrt(distance)
    else:
        return float('nan')


def get_neighbors(row, train_data, k):
    dists = []

    size = len(train_data) - 1

    for i in range(len(test_data)):
        dist = euclidean_distance(train_data[i], row)
        dists.append((train_data[i], dist))

    dists.sort(key=lambda x: x[1])

    neighbors = []
    for i in range(k):
        neighbors.append(dists[i][0])

    return neighbors


def majority_lang(neighbors):
    langs = {}

    for i in range(len(neighbors)):
        lang = neighbors[i][-1]
        if lang in langs:
            langs[lang] += 1
        else:
            langs[lang] = 1

    sorted_langs = sorted(langs.iteritems(), key=lambda x: x[1])
    return sorted_langs[-1][0]


def accuracy(test_data, classifications):
    correct_count = 0.0
    for i in range(len(test_data)):
        if test_data[i][-1] is classifications[i]:
            correct_count += 1

    return correct_count / float(len(test_data))


train_data = []
test_data = []
classifications = []
k = 3

read_data(train_data, test_data)

for row in test_data:
    neighbors = get_neighbors(row, train_data, k)

    result = majority_lang(neighbors)

    print row[0], result

    classifications.append(result)


