"""
Adapted from API.py
"""

import sys
import os.path
import pickle
import numpy as np
import csv
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.metrics.pairwise import *
from joblib import dump, load
import word2vec


def evaluation_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
    
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    return acc, prc, rc, f1, auc_

def bfp_clf_results(train_data, labels, algorithm=None, kfold=5,sample_weight=None):
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    skf = StratifiedKFold(n_splits=kfold,shuffle=True)
    print('Algorithm results:', algorithm)
    accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
    index = [[train, test] for train, test in skf.split(train_data, labels)]

    train_index, test_index = index[0][0], index[0][1]
    x_train, y_train = train_data[train_index], labels[train_index]
    weight = sample_weight[train_index]

    x_test, y_test = train_data[test_index], labels[test_index]
    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs', max_iter=1000,tol=0.2).fit(X=x_train, y=y_train)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto', probability=True, kernel='linear',class_weight='balanced',max_iter=10,
                  tol=0.1).fit(X=x_train, y=y_train)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=x_train, y=y_train)
    elif algorithm == 'dt':
        clf = DecisionTreeClassifier().fit(X=x_train, y=y_train,sample_weight=None)
    y_pred = clf.predict_proba(x_test)[:, 1]
    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)
# accs.append(acc)
    # prcs.append(prc)
    # rcs.append(rc)
    # f1s.append(f1)
    # aucs.append(auc_)
    return acc, prc, rc, f1, auc_

def train_model(train_data, labels, algorithm=None,):
    print('training ... ')
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    print('Algorithm:', algorithm)
    x_train, y_train = train_data, labels

    clf = None
    if algorithm == 'lr':
        clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)
    elif algorithm == 'svm':
        clf = SVC(gamma='auto', probability=True, kernel='linear',class_weight='balanced',max_iter=1000,
                  tol=0.1).fit(X=x_train, y=y_train)
    elif algorithm == 'nb':
        clf = GaussianNB().fit(X=x_train, y=y_train)
    elif algorithm == 'dt':
        clf = DecisionTreeClassifier().fit(X=x_train, y=y_train,sample_weight=None)

    dump(clf, '../data/model/{}_bert.joblib'.format(algorithm))
    return


def learn_embedding(embedding_method, patch_txt):
    w = Word2vector(embedding_method)
    try:
        patch_vector = w.embedding(patch_txt)
    except Exception as e:
        print(e)
        raise e
    return patch_vector


def predict(x_test, algorithm):
    threshold = 0.5
    print('loading model ... ')
    clf = load('../data/model/{}_bert.joblib'.format(algorithm))

    x_test = np.array(x_test).reshape((1,-1))
    y_pred_prob = clf.predict_proba(x_test)[0,1]
    y_pred = 1 if y_pred_prob >= threshold else 0

    return y_pred, y_pred_prob

def process_patches_directory(directory_path, algorithm, output_csv=None):
    correct_dir = os.path.join(directory_path, 'correct')
    overfitting_dir = os.path.join(directory_path, 'overfitting')
    
    if not os.path.exists(correct_dir) or not os.path.exists(overfitting_dir):
        raise Exception(f"Please ensure that '{directory_path}' contains 'correct' and 'overfitting' subdirectories.")
    
    y_true = []
    y_pred = []
    patch_results = []  # to store [patch_name, actual, prediction]
    
    # Process correct patches (labeled as 1)
    print(f"Processing correct patches from {correct_dir}...")
    for filename in os.listdir(correct_dir):
        file_path = os.path.join(correct_dir, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as f:
                    patch_txt = f.read()
                    x_test_vector, _ = word2vec.learned_feature(patch_txt, 'bert')
                    prediction, _ = predict(x_test_vector, algorithm)
                    y_true.append(1)
                    y_pred.append(prediction)
                    patch_results.append([filename, 'correct', 'correct' if prediction == 1 else 'overfitting'])
                    print(f"Processed {filename}: {'CORRECT' if prediction == 1 else 'INCORRECT'}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Process overfitting patches (labeled as 0)
    print(f"Processing overfitting patches from {overfitting_dir}...")
    for filename in os.listdir(overfitting_dir):
        file_path = os.path.join(overfitting_dir, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as f:
                    patch_txt = f.read()
                    x_test_vector, _ = word2vec.learned_feature(patch_txt, 'bert')
                    prediction, _ = predict(x_test_vector, algorithm)
                    y_true.append(0)
                    y_pred.append(prediction)
                    patch_results.append([filename, 'overfitting', 'correct' if prediction == 1 else 'overfitting'])
                    print(f"Processed {filename}: {'CORRECT' if prediction == 1 else 'INCORRECT'}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Write results to CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['patch_name', 'correctness', 'prediction'])
            writer.writerows(patch_results)
        print(f"Results written to {output_csv}")
    
    # Calculate metrics for return value (still useful for analysis)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    # Calculate recall for positive class (correct patches)
    pos_recall = recall_score([y for y in y_true if y == 1], 
                             [y_pred[i] for i, y in enumerate(y_true) if y == 1], 
                             zero_division=0) if 1 in y_true else 0
    
    # Calculate recall for negative class (overfitting patches)
    neg_recall = recall_score([1-y for y in y_true if y == 0], 
                             [1-y_pred[i] for i, y in enumerate(y_true) if y == 0], 
                             zero_division=0) if 0 in y_true else 0
    
    # Calculate F1 score
    f1 = 2 * precision * pos_recall / (precision + pos_recall) if (precision + pos_recall) > 0 else 0
    
    return accuracy, precision, pos_recall, neg_recall, f1


def get_feature(buggy, patched):
    return subtraction(buggy, patched)

def get_features(buggy, patched):

    subtract = subtraction(buggy, patched)
    multiple = multiplication(buggy, patched)
    cos = cosine_similarity(buggy, patched).reshape((-1,1))
    euc = euclidean_similarity(buggy, patched).reshape((-1,1))

    fe = np.hstack((subtract, multiple, cos, euc))
    return fe

def subtraction(buggy, patched):
    return buggy - patched

def multiplication(buggy, patched):
    return buggy * patched

def cosine_similarity(buggy, patched):
    return paired_cosine_distances(buggy, patched)

def euclidean_similarity(buggy, patched):
    return paired_euclidean_distances(buggy, patched)


if __name__ == '__main__':
    # Start timing
    start_time = time.time()
    
    directory_path = '../../../patches_by_time/patches_8h_deduplicated'

    model = 'bert'
    algorithm = 'dt'

    print('model: {}'.format(model))

    path = '../data/experiment3/kui_data_for_' + model + '.pickle'

    with open(path, 'rb') as input:
        data = pickle.load(input)
    label, buggy, patched = data

    index_p = list(np.where(label == 1)[0])
    index_n = list(np.where(label == 0)[0])

    # data
    train_data = get_features(buggy, patched)

    # train
    if not os.path.exists('../data/model/{}_bert.joblib'.format(algorithm)):
        print('Need to train')
        train_model(train_data=train_data, labels=label, algorithm=algorithm)

    # Generate CSV output path and process patches
    output_csv = f"patch_results_{algorithm}.csv"
    process_patches_directory(directory_path, algorithm, output_csv)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    minutes, seconds = divmod(execution_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    # Format the time string
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)} hours, "
    if minutes > 0 or hours > 0:
        time_str += f"{int(minutes)} minutes, "
    time_str += f"{seconds:.2f} seconds"
    
    print(f"Results successfully saved to {output_csv}")
    print(f"Total execution time: {time_str}")
