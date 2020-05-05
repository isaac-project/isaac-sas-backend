# -*- coding: utf-8 -*-
"""
Bag-of-words benchmark for IQB TBA data

Based on ASAP-SAS BOW benchmark by Ben Hamner (https://github.com/benhamner/ASAP-SAS).
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import re
import numpy
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import iqb_io
import textdistance as td
import sys
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn import svm


TARGET_DELIMS_PATTERN = re.compile('|'.join(map(re.escape, ["•","ODER", " / "])))
TEST_DATA_PATH = 'data/test/'
TRAIN_DATA_PATH = 'data/train/'
filename_test = TEST_DATA_PATH + '{}_{}_test.tsv'
filename_train = TRAIN_DATA_PATH + '{}_{}_train.tsv'

MEASURES = [
        td.algorithms.edit_based.NeedlemanWunsch(),
        td.algorithms.edit_based.SmithWaterman(),
        td.algorithms.sequence_based.LCSSeq(),
        td.algorithms.sequence_based.LCSStr(),
        td.algorithms.simple.Length(),
        td.algorithms.phonetic.Editex(),
        td.algorithms.phonetic.MRA(),
        td.algorithms.token_based.Overlap(),
        td.algorithms.token_based.Cosine()
        ]

bow_scores = {}
sim_scores = {}
df = iqb_io.read_iqb_data()
varinfo = iqb_io.read_varinfo("data/original_data/varinfo.tsv")

answers_with_varinfo = pd.merge(df, varinfo, how='left', on=[iqb_io.VAR_COLUMN])
df = answers_with_varinfo

simCache = {}


def train_bag(text, n=500):
    words = [w for w in text.lower().split(" ") if w]
    word_counts = {}
    for w in words:
        if w not in word_counts:
            word_counts[w] = 0.0
        word_counts[w] += 1.0

    sorted_words = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)
    return sorted_words[:n]

def bag_representation(bag, text):
    return [float(w in str(text)) for w in bag]

def bag_count_representation(bag, text):
    return [float(len(re.findall(w, text))) for w in bag]

def get_target_alternatives(targetstr):

    return TARGET_DELIMS_PATTERN.split(targetstr)

def sim_lookup_str(response, a, m):
    return response + " | " + a + " | " + type(m).__name__

def similarity_features(response, target, measures):
    target = str(target)
    response = str(response)
    resultScores = []
    alternatives = get_target_alternatives(target)
    
    for m in measures:
        max_sim = -1.0
        for a in alternatives:
            a = str(a)
            try:
                lookup = sim_lookup_str(response, a, m)
                simCache.setdefault(lookup, m.normalized_similarity(response,a))
                sim = simCache[lookup]
                if sim > max_sim:
                    max_sim = sim
            except:
                # this should only happen for German answers and phonetic distance measures
                print("Error with measure", m, "on strings '", response, "' and '", a, "'", file=sys.stderr)

        resultScores.append(max_sim)
    
    return resultScores

def bow_features(bag, instances):
    return [bag_representation(bag, x) for x in instances["value.raw"]]

def bow_classifier(bag, train_instances: pd.DataFrame, y, clf,  n_folds=3, crossval=False):
    fea = bow_features(bag, train_instances)
    if crossval:
        try:
            return cross_val_score(clf, fea, y, cv=n_folds)
        except ValueError as e:
            print(e.__str__())
            print("CONTINUED ONTO THE NEXT ONE")
            return None
    clf.fit(fea,y)
    return clf

def majority_baseline(instances):
    counts = instances.value_counts()
    keys = counts.keys().tolist()
    values = counts.tolist()
    return (keys[0], values[0]/len(instances))


def split_train_test(test_data_size = 0.4):
    """
        Splits a pandas data frame with instances into train and test files.
        The training files are written to data-analysis/data/train while the test files
        are written to data-analysis/data/test. The name of the files follows this pattern:
        taskId_taskField_train.tsv and taskId_taskField_test.tsv. TaskId is the id of the main task, taskField is the id
        of a subtask. Ex. If taskId ABC123 has three subtask a) b) and c), the names of the training files will be
         ABC123_ABC123a_train.tsv, ABC123_ABC123b_train.tsv, ABC123_ABC123c_train.tsv
    :param test_data_size: size of the test data set (default = 0.4)

    """

    print("Number of tasks: ", len(df["variable_codebook"].unique()))
    for item in df["variable_codebook"].unique():
        all_instances_per_task = df[df["variable_codebook"] == item].dropna()
        distrib = all_instances_per_task["value.coded"].astype("category").value_counts()
        print("Class Distribution:\n", distrib)
        if distrib.tolist()[-1] == 1:
            singleton = distrib.keys().tolist()[-1]
            print("Removing singleton instance of class:", singleton)
            indexNames = all_instances_per_task[all_instances_per_task['value.coded'] == singleton].index

            all_instances_per_task.drop(indexNames, inplace=True) # todo: keep singletons and add to train?

        # split data into train and test data
        y = all_instances_per_task["value.coded"]
        X_train,X_test,y_train,y_test = train_test_split(all_instances_per_task, y,test_size=test_data_size, stratify=y)
        task_id = X_train['task'].unique()[0]

        # write training and test instances to separate files
        X_train.to_csv(filename_train.format(task_id, item), encoding='utf-8', index= False, sep='\t')
        X_test.to_csv(filename_test.format(task_id,item), encoding='utf-8', index= False, sep='\t')

def load_data(filename):
    return pd.read_csv(filename, encoding='utf-8',  sep='\t', dtype= {'variable': 'category',
                            'ID': 'category',
                            'GroupVar': 'category',
                            'Booklet': 'category',
                            'value.raw': 'str',
                            'value.coded': 'float64',
                            'domain': 'category',
                            'task': 'category',
                            'value.numWords': 'int64',
                                                              'variable_codebook': 'category',
                                                              'Unnamed: 0': 'str',
                                                              'prompt': 'str',
                                                              'targets_value': 'int',  # todo: change to string?
                                                              'targets': 'str',
                                                              'non_targets': 'str',
                                                              })

def get_test_scores():
    bow_scores = {}
    sim_scores = {}
    training_files = [join(TRAIN_DATA_PATH, f) for f in listdir(TRAIN_DATA_PATH) if isfile(join(TRAIN_DATA_PATH, f))]


def get_crossval_scores_per_task(bow_clf, sim_clf, n_folds = 3):
    """
    Performs per task cross-validation on the training data.
      :param n_folds: The number of folds. The default is 3
      :return:
    """
    cros_val = True
    bow_crossval_scores = {}
    sim_crossval_scores = {}
    training_files = [join(TRAIN_DATA_PATH, f) for f in listdir(TRAIN_DATA_PATH) if isfile(join(TRAIN_DATA_PATH, f))]

    for f in training_files:
        print("Training file name: ", f)
        training_data = load_data(f)
        training_labels = training_data['value.coded']
        task_field = training_data['variable_codebook'][0]

        # Remove the labels from df
        training_data.pop('value.coded')

        # BOW
        bag = train_bag(" ".join(training_data["value.raw"]), 500)
        bow_crosval = bow_classifier(bag, training_data, training_labels, bow_clf, n_folds, cros_val)
        if bow_crosval is not None:
            bow_crossval_scores[task_field] = numpy.mean(bow_crosval)
        else:
            bow_crossval_scores[task_field] = None

        # SIM
        sim_feats_train = [similarity_features(r, t, MEASURES) for r, t
                           in zip(training_data["value.raw"], training_data["targets"])]
        try:
            sim_crossval = cross_val_score(sim_clf, sim_feats_train, training_labels, cv=n_folds)
            sim_crossval_scores[task_field] = numpy.mean(sim_crossval)
        except ValueError as e :
            print(e.__str__())
            sim_crossval_scores[task_field] = None

        print("Average SIM cross-val score for task {} : ".format(task_field), sim_crossval_scores[task_field])

        print("Average BOW cross-val score for task {}: ".format(task_field), bow_crossval_scores[task_field])
        print("--------------------------------------------------")

    return bow_crossval_scores, sim_crossval_scores


def main():
    for item in df["variable_codebook"].unique():
        print("Making Predictions for item: %s" % item)
        all_instances = df[df["variable_codebook"] == item].dropna()
        distrib = all_instances["value.coded"].astype("category").value_counts()
        
        print("Class Distribution:\n", distrib)
        if distrib.tolist()[-1] == 1:
            singleton = distrib.keys().tolist()[-1]
            print("Removing singleton instance of class:", singleton)
            indexNames = all_instances[all_instances['value.coded'] == singleton].index
            all_instances.drop(indexNames, inplace=True)
        
        y = all_instances.pop("value.coded")
        print("Majority:", majority_baseline(y))
        X_train,X_test,y_train,y_test = train_test_split(all_instances,y,test_size=0.4,stratify=y)
        print(X_train)
        print(y_train)
        #train_instances = all_instances.sample(frac=0.6)
        #print("Train Majority:", majority_baseline(y_train))
        
        #test_instances = all_instances.drop(train_instances.index)
        #print("Test Majority:", majority_baseline(y_test))
        
        # BOW
        bag = train_bag(" ".join(X_train["value.raw"]), 500)
        clf = bow_classifier(bag, X_train, y_train)
        fea = bow_features(bag, X_test)
        
        bow_score = clf.score(fea, y_test)

        print("BOW Score:", bow_score)
        bow_scores[item] = bow_score

        # SIM
        sim_feats_train = [similarity_features(r,t,MEASURES) for r,t
                           in zip(X_train["value.raw"], X_train["targets"])]
        sim_clf = rf()
        sim_clf.fit(sim_feats_train, y_train)
        sim_feats_test = [similarity_features(r,t,MEASURES) for r,t
                           in zip(X_test["value.raw"], X_test["targets"])]
        sim_score = sim_clf.score(sim_feats_test, y_test)
        print("SIM Score:", sim_score)
        sim_scores[item] = sim_score
        print()

def another_main():
    #for classifier in Classifier:
    #bow_classifiers = {'rf': rf()}
    #sim_classifiers = {'rf': rf()}
    bow_classifiers = {'log': LogisticRegression(),  'svc': svm.SVC()} #'rf': rf(),
    sim_classifiers = {'log': LogisticRegression(), 'svc': svm.SVC()} #'rf': rf(),
    final_scores = {}
    for classifier in bow_classifiers.keys():
        bow_model = bow_classifiers[classifier]
        sim_model = sim_classifiers[classifier]
        bow_scores, sim_scores = get_crossval_scores_per_task(bow_model,
                                                              sim_model, 5)
        #def get_crossval_scores_per_task(bow_clf, sim_clf, n_folds = 3):
        final_scores[classifier+ "_bow"] = [(k, v) for k, v in bow_scores.items()]
        final_scores[classifier + "_sim"] = [(k, v) for k, v in sim_scores.items()]
    for i, v in final_scores.items():
        print(i, v)

    with open("table_with_results.txt", "w") as f:
        for i, v in final_scores.items():
            f.write("----------------------")
            f.write(i)
            f.write("----------------------")
            for k in v:
                k = k[0] + "\t" + str(k[1])
                f.write(k)

if __name__=="__main__":
    #main()
    #split_train_test()
    #b,s = get_crossval_scores_per_task()
    #print(b)
    #print(s)
    another_main()
