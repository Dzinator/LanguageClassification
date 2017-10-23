'''
    Binary Decision implementation

    TODO cross validation within training data

    Yanis
'''

import random
import csv
import pickle
# import knn


def read_data(train_data=[], test_data=[], from_train=False):
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


def get_gini(groups, labels):
    n_rows = float(sum([len(group) for group in groups]))

    gini = 0.0

    for group in groups:
        size = float(len(group))
        if size > 0:
            score = 0.0
            for label in labels:
                update = [row[-1] for row in group].count(label) / size
                score = score + (update * update)

            gini += (1.0 - score) * (size / n_rows)
        else:
            continue

    return gini


def split(feature_index, data):
    left, right = list(), list()
    for row in data:
        #Split if contains letter
        if row[feature_index] > 0:
            left.append(row)
        else:
            right.append(row)
    return left, right


def best_split(data):
    labels = list(set(row[-1] for row in data))

    #best parameters will be saved in these
    b_fi = float('inf')
    b_score = float('inf')
    b_groups = None

    #iterate on all special characters features
    if len(data) > 0:
        for i in range(27, len(data[0])-1):
            for row in data:
                groups = split(i, data)
                gini = get_gini(groups, labels)
                if gini < b_score:
                    b_fi, b_score, b_groups = i, gini, groups

    return {'index': b_fi, 'groups': b_groups}


def predict(node, row):
    if row[node['index']] > 0:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def predict_knn(node, row):
    #TODO
    pass

def majority_label(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)


def make_children(node, max_d, min_s, d):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        #terminal
        node['left'] = node['right'] = majority_label(left + right)
        return

    if d >= max_d:
        node['left'], node['right'] = majority_label(left), majority_label(right)
        return

    if len(left) <= min_s:
        node['left'] = majority_label(left)
    else:
        node['left'] = best_split(left)
        make_children(node['left'], max_d, min_s, d+1)

    if len(right) <= min_s:
        node['right'] = majority_label(right)
    else:
        node['right'] = best_split(right)
        make_children(node['right'], max_d, min_s, d+1)


def tree(data, max_depth, min_size):
    root = best_split(data)
    make_children(root, max_depth, min_size, 1)
    return root






train_data = []
test_data = []
classifications = []
k = 3
max_depth = 25
min_group_size = 12

read_data(train_data, test_data, True)

print len(train_data)
print len(test_data)


#save/load tree

try:
    with open('tree.bin', 'r') as fp:
        tree = pickle.loads(fp.read())
    print '-Loaded pickeld data'
except:
    tree = tree(train_data, 25, 12)
    with open('tree.bin', 'w') as fp:
        fp.write(pickle.dumps(tree))

accurate = 0
with open('bdt.csv', 'wb') as f:
    csv_writer = csv.writer(f)

    csv_writer.writerow(['Id', 'Category'])
    for row in test_data:
        prediction = predict(tree, row)

        csv_writer.writerow([row[0], prediction])

        # if row[-1] == prediction:
        #     accurate += 1


print('Predicted %d out of %d' % (accurate, len(test_data)))
