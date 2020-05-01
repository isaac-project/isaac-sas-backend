#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:13:18 2020

@author: rziai
"""

import pandas as pd

TASK_STR = "Aufgabe:"
VARIABLE_STR = "Variable:"
PROMPT_STR = "Allgemein:"
CORRECT_STR = "RICHTIG"
INCORRECT_STR = "FALSCH"

VAR_COLUMN = 'variable_codebook'
PROMPT_COLUMN = 'prompt'
SCORE_COLUMN = 'targets_value'
TARGET_COLUMN = 'targets'
NON_TARGET_COLUMN = 'non_targets'

def read_iqb_data(filename: str = "iqb-tba-answers-doublequotes.tsv") -> pd.DataFrame: #"/home/rziai/git/iqb-tba-data/iqb-tba-answers.tsv") -> pd.DataFrame:
    df = pd.read_csv(filename,
                     dtype={'variable': 'category',
                            'ID': 'category',
                            'GroupVar': 'category',
                            'Booklet': 'category',
                            'value.raw': 'str',
                            'value.coded': 'float64',
                            'domain': 'category',
                            'task': 'category',
                            'value.numWords': 'int64'},
                             encoding='utf8', sep= "\t") #, quotechar="\"", escapechar="\\")
    df["variable_codebook"] = [s.split("_")[0] for s in df["variable"]]
    df["variable_codebook"] = df["variable_codebook"].astype("category")
    return df

def parse_variable_data(filename: str = "/home/rziai/git/iqb-tba-data/BT fuer UT/variable_info.txt") -> pd.DataFrame:
    vardata = {}
    vardata[VAR_COLUMN] = []
    vardata[PROMPT_COLUMN] = []
    vardata[SCORE_COLUMN] = []
    vardata[TARGET_COLUMN] = []
    vardata[NON_TARGET_COLUMN] = []
    
    in_variable = False
    in_targets = False
    in_non_targets = False
    current_target_strs = []
    current_non_target_strs = []

    with open(filename, encoding='utf-8') as infile:
        for line in infile.readlines():
            if line.startswith(TASK_STR):
                continue
            if line.startswith(VARIABLE_STR):
                vardata[VAR_COLUMN].append(line[len(VARIABLE_STR):].strip())
                in_variable = True
                if in_non_targets:
                    vardata[NON_TARGET_COLUMN].append(" ".join(current_non_target_strs))
                    current_non_target_strs = []
                    in_non_targets = False
            elif line.startswith(PROMPT_STR):
                vardata[PROMPT_COLUMN].append(line[len(PROMPT_STR):].strip())
            elif line.startswith(CORRECT_STR):
                in_targets = True
            elif in_variable and line.startswith(INCORRECT_STR):
                vardata[SCORE_COLUMN].append(current_target_strs[0])
                vardata[TARGET_COLUMN].append(" ".join(current_target_strs[1:]))
                current_target_strs = []
                # pad prompts in case there is a mismatch
                if len(vardata[PROMPT_COLUMN]) < len(vardata[VAR_COLUMN]):
                    vardata[PROMPT_COLUMN].append(None)
                in_targets = False
                in_variable = False
                in_non_targets = True
            elif in_targets:
                current_target_strs.append(line.strip())
            elif in_non_targets:
                current_non_target_strs.append(line.strip())


    vardata[NON_TARGET_COLUMN].append(" ".join(current_non_target_strs))
    print(vardata)
    return pd.DataFrame.from_dict(vardata)

def merge():
    df = read_iqb_data()
    varinfo = pd.read_table("varinfo.tsv")  # ("/home/rziai/git/iqb-tba-data/varinfo.tsv")

    answers_with_varinfo = pd.merge(df, varinfo, how='left', on=[VAR_COLUMN])
    return answers_with_varinfo

def write_to_csv(filename):
    dataframe = merge()
    dataframe.to_csv(filename, encoding='utf-8', index_label= 'rownum', sep='\t')
