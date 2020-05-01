# -*- coding: utf-8 -*-
"""
Bag-of-words benchmark for IQB TBA data

Based on ASAP-SAS BOW benchmark by Ben Hamner (https://github.com/benhamner/ASAP-SAS).
"""

import pandas as pd
import re
import numpy
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import train_test_split
import iqb_io
import textdistance as td
import sys
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import cross_val_score
from statistics import mean

TARGET_DELIMS_PATTERN = re.compile('|'.join(map(re.escape, ["•","ODER", " / "])))
TEST_DATA_PATH = 'data/test/{}/'
TRAIN_DATA_PATH = 'data/train/{}/'
filename_test = TEST_DATA_PATH + "{}_test.tsv"
filename_train = TRAIN_DATA_PATH + "{}_train.tsv"

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
varinfo = pd.read_table("varinfo.tsv") #("/home/rziai/git/iqb-tba-data/varinfo.tsv")

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
    # TODO: when text = int, it can't iterate over it
    return [float(w in str(text)) for w in bag]

def bag_count_representation(bag, text):
    return [float(len(re.findall(w, text))) for w in bag]

def get_target_alternatives(targetstr):
    #print(targetstr, type(targetstr))
    return TARGET_DELIMS_PATTERN.split(targetstr)

def sim_lookup_str(response, a, m):
    return response + " | " + a + " | " + type(m).__name__

def similarity_features(response, target, measures):
    
    resultScores = []
    target = str(target) #TODO added this
    alternatives = get_target_alternatives(target)
    
    for m in measures:
        max_sim = -1.0
        for a in alternatives:
            response = str(response) # TODO: added this
            a = str(a)
            #print(response, type(response))
            #print(a, type(a))
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

def bow_classifier(bag, train_instances: pd.DataFrame, y, crosval=False):
    fea = bow_features(bag, train_instances)
    clf = rf()
    if crosval:
        return cross_val_score(clf, fea, y, cv=5)
    clf.fit(fea,y)
    return clf

def majority_baseline(instances):
    counts = instances.value_counts()
    keys = counts.keys().tolist()
    values = counts.tolist()
    return (keys[0], values[0]/len(instances))



def split_train_test():
    filename_test = TEST_DATA_PATH + "{}_test.tsv"
    filename_train = TRAIN_DATA_PATH + "{}_train.tsv"
    print(len(df["variable_codebook"].unique()))
    for item in df["variable_codebook"].unique():
        all_instances_per_task = df[df["variable_codebook"] == item].dropna()
        distrib = all_instances_per_task["value.coded"].astype("category").value_counts()
        print("Class Distribution:\n", distrib)
        if distrib.tolist()[-1] == 1:
            singleton = distrib.keys().tolist()[-1]
            print("Removing singleton instance of class:", singleton)
            indexNames = all_instances_per_task[all_instances_per_task['value.coded'] == singleton].index
            all_instances_per_task.drop(indexNames, inplace=True) # todo: keep singletons and add to train?

        y = all_instances_per_task["value.coded"]
        X_train,X_test,y_train,y_test = train_test_split(all_instances_per_task, y ,test_size=0.4, stratify=y)
        X_train.to_csv(filename_train.format(item), encoding='utf-8', index= False, sep='\t')
        X_test.to_csv(filename_test.format(item), encoding='utf-8', index= False, sep='\t')

def load_data(filename):
    return pd.read_csv(filename, encoding='utf-8',  sep='\t')

def main_demo():
    cros_val = True
    bow_scores = {}
    sim_scores = {}
    bow_crossval_scores = {}
    sim_crossval_scores = {}
    training_files = [join(TRAIN_DATA_PATH, f) for f in listdir(TRAIN_DATA_PATH) if isfile(join(TRAIN_DATA_PATH, f))]
    testing_files = [join(TRAIN_DATA_PATH, f) for f in listdir(TEST_DATA_PATH) if isfile(join(TEST_DATA_PATH, f))]

    for f in training_files:
        print("Training file name: ", f)
        training_data = load_data(f)
        training_labels = training_data['value.coded']

        task_id = training_data['variable_codebook'][0]

        test_data = load_data(f.replace('train', 'test'))
        test_labels = test_data['value.coded']

        #print("Printing training data: ")
        #print(training_data)

        #print("Printing training labels: ")
        #print(training_labels)

        # Remove the labels from df
        training_data.pop('value.coded')
        test_data.pop('value.coded')


        # BOW
        bag = train_bag(" ".join(training_data["value.raw"]), 500)
        bow_crosval = bow_classifier(bag, training_data, training_labels, cros_val)
        bow_crossval_scores[task_id] = numpy.mean(bow_crosval)

        clf = bow_classifier(bag, training_data, training_labels)
        fea = bow_features(bag, test_data)

        bow_score = clf.score(fea, test_labels)
        bow_scores[task_id] = bow_score


        #print("BOW Score for {} :".format(task_id), bow_score)

        # SIM
        sim_feats_train = [similarity_features(r, t, MEASURES) for r, t
                           in zip(training_data["value.raw"], training_data["targets"])]
        sim_clf = rf()
        sim_crossval = cross_val_score(sim_clf, sim_feats_train, training_labels, cv=5)
        sim_crossval_scores[task_id] = numpy.mean(sim_crossval)


        sim_clf.fit(sim_feats_train, training_labels)
        sim_feats_test = [similarity_features(r, t, MEASURES) for r, t
                          in zip(test_data["value.raw"], test_data["targets"])]

        sim_score = sim_clf.score(sim_feats_test, test_labels)

        #print("SIM score for {} :".format(task_id), sim_score)
        sim_scores[task_id] = sim_score
        #print()
    print("All BOW scores:")
    print(bow_scores)
    print("All SIM scores:")
    print(sim_scores)
    print("Crossval BOW scores:")
    print(bow_crossval_scores)
    print("Crossval SIM scores:")
    print(sim_crossval_scores)
    print("Average bow score over {} tasks: ".format(len(bow_scores)), mean(bow_scores.values()))
    print("Average sim score over {} tasks: ".format(len(sim_scores)), mean(sim_scores.values()))
    print("Printing av. SIM cross-val score: ", mean(sim_crossval_scores.values()))
    print("Printing av. BOW cross-val score: ", mean(bow_crossval_scores.values()))

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


if __name__=="__main__":
    #main()
    #split_train_test()
    main_demo()
