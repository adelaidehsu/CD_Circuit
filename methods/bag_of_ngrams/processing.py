# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import sklearn
import random

from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from bpe import Encoder


from pyfunctions.general import *
    
"""
* Characters to strip from pathology reports
"""
stripChars = ['\"', ',', ';', 'null', '*', '#', '~', '(', ')' ,"\"", '\'']

def getCounter(data):
    """
    * Given a list of tuples containing the mrn, accession id, and report
    * as strings, return a Python counter on the concatenated reports
    """
    lst = []
    for patient in data:
        doc = patient['clean_document']
        lst = lst + doc.split()
    counter = Counter(lst)
    return counter

def cleanReport(report, chars, include_dash=False, periods=True):
    """
    * Clean a single report given as a string and return the result
    """
    report = report.lower()
    for c in chars:
        report = report.replace(c, ' ')

    """
    * Replace / and = characters with spaces preceding and following it
    """
    for c in ['+','/', '=' , ':', '(', ')','<','>']:
        report = report.replace(c, ' '+ c + ' ')
    if include_dash:
        report = report.replace('-', ' - ')

    """
    * Remove periods
    """
    processed = []
    for token in report.split():
        
        if periods:
            token = token.rstrip('.')
            token = token.strip()
            token = token.rstrip('.')
            
        if token == 'i':
            token = '1'
        elif token == 'ii':
            token = '2'
        elif token == 'iii':
            token = '3'
        processed.append(token)
    processed = " ".join(processed)

    return " ".join(processed.split())

def cleanReports(patients, stripObjects,include_dash=False, periods=True):
    """
    * Given a list of tuples containing the mrn, accession id, and report
    * clean each report and return the processed list of tuples
    """
    report_list = []
    for i, patient in enumerate(patients):
        clean_report = cleanReport(patient['document'], stripObjects,include_dash=include_dash, periods=periods)
        patients[i]['clean_document'] = clean_report
        report_list.append(clean_report)

    return patients

def cleanSplit(data, stripObjects, include_dash=False, periods=True):
    """ 
    * Given a list of the corpus split (training, validation and test)
    * clean all the reports within each split and return the list
    """
    splits = ['train','val','test']
    

    for split in splits:
        data[split] = cleanReports(data[split], stripObjects, include_dash, periods=periods)
            
    return data

def unkReport(report,counter):
    processed = []
    for token in report.split():
        if counter[token] < 2:
            processed.append("<unk>")
        else:
            processed.append(token)
    processed = " ".join(processed)
    return processed  

def unkReports(data, counter):    
    """ 
    * Given a corpus as a list of tuples containing the mrn, acc, and
    * report (as string), replace all the rare words (occuring less than 
    * 2 times) with unk and return the processed corpus 
    """
    processed_reports = []
    for i, patient in enumerate(data):
        report = patient['clean_document']
        processed = []
        for token in report.split():
            if counter[token] <= 2:
                processed.append("<unk>")
            else:
                processed.append(token)
        processed = " ".join(processed)
        patient['clean_document_unked'] = processed
        data[i] = patient
    return data

def getTrainedVectorizer(corpus, N, min_n):
    """
    * Return a trained sklearn CountVectorizer on a given corpus (list of 
    * strings) and a specified N (Ngram)
    """
    textVectorizer = CountVectorizer(stop_words=None,
                                tokenizer=TreebankWordTokenizer().tokenize, 
                                ngram_range = (min_n,N))
    textVectorizer.fit(corpus)
    return textVectorizer