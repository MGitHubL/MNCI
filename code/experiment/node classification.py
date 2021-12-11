from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import lil_matrix
import numpy as np
import json


def format_training_data_for_dnrl(emb_file, i2l_file):
    i2l = dict()
    with open(i2l_file, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            i2l[n_id] = l_id

    i2e = dict()
    with open(emb_file, 'r') as reader:
        reader.readline()
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_id = embeds[0]
            if node_id in i2l:
                i2e[node_id] = embeds[1:]
                
    Y = []
    X = []
    i2l_list = sorted(i2l.items(), key=lambda x: x[0])
    for (the_id, label) in i2l_list:
        Y.append(label)
        X.append(i2e[the_id])

    X = np.stack(X)
    return X, Y


def lr_classification(X, Y, cv):
    # clf = KNeighborsClassifier()
    clf = LogisticRegression()
    scores_1 = cross_val_score(clf, X, Y, cv=cv, scoring='f1_micro', n_jobs=8)
    scores_2 = cross_val_score(clf, X, Y, cv=cv, scoring='f1_weighted', n_jobs=8)
    scores_1 = scores_1.sum() / 5
    scores_2 = scores_2.sum() / 5
    return scores_1, scores_2


if __name__ == '__main__':
    X, Y = format_training_data_for_dnrl('../emb/bitalpha/bitalpha_MNCI_5.emb', '../../data/bitalpha/node2label.txt')
    accuracy_score, weighted_score = lr_classification(X, Y, cv=5)
    print(accuracy_score)
    print(weighted_score)
