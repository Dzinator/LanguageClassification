'''
    K-Nearest Neighbors implementation
    
    TODO cross validation within training data
    
    Yanis
'''

import csv
import random
import pickle


def extract_data(train_data=[], test_data=[], from_train=False):
    # restore/save processed train data
    try:
        with open('sanitized_pickle.bin', 'r') as fp:
            train_data = pickle.loads(fp.read())
        print '-Loaded pickeld data'
    except:
        with open('id_vector_train.csv', 'rb') as data, \
                open('train_set_y.csv', 'rb') as label_data:
            lines = csv.reader(data)
            labels = csv.reader(label_data)
            # skip headers
            next(lines)
            next(labels)

            count = 0
            for line, label in zip(lines, labels):
                row = [int(float(x)) for x in (line + [label[-1]])]

                count += 1

                if from_train:
                    if count > 200:
                        break
                    if random.random() < 0.8:
                        train_data.append(row)
                    else:
                        test_data.append(row)
                else:
                    train_data.append(row)

        with open('sanitized_pickle.bin', 'w') as fp:
            fp.write(pickle.dumps(train_data))


    if not from_train:
        with open('id_vector_test.csv', 'rb') as csv_data:
            lines = csv.reader(csv_data)

            next(lines)

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

        return distance
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


train_data = []
test_data = []
classifications = []
k = 3

extract_data(train_data, test_data, True)

accurate = 0
for row in test_data:
    neighbors = get_neighbors(row, train_data, k)

    result = majority_lang(neighbors)

    if row[-1] == result:
        accurate += 1

    classifications.append(result)

print('Predicted %d out of %d' % (accurate, len(test_data)))