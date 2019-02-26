import os
import string
import re

from nltk.tokenize import word_tokenize
import sklearn

from utils import fb_top_intent

class DataCleaner(object):

    def __init__(self, tokenization_type, dataset='fb'):
        self.token_type = tokenization_type 
        self.dataset_type = dataset

    def filterOutPunctuation(self, word):
        punctuationDict = {ch: '' for ch in string.punctuation}
        reformedWord = ""
        for ch in word:
            if ch in punctuationDict:
                reformedWord += punctuationDict[ch]
            else:
                reformedWord += ch
        return reformedWord

    def wordTokenize(self, line):
        '''
        token_type = 1: Simple split
        token_type = 2: nltk.token 
        '''
        if self.token_type == 1:
            return line.split()
        elif self.token_type == 2:
            return word_tokenize(line)
        else:
            return line.split()

    def applyRegEx(self, line):
        regex = "([@][A-Za-z0-9]+)|([^0-9A-Za-z# \t])|(\w+:\/\/\S+)|(#[^A-Za-z0-9]+)"
        reg_line = re.sub(regex, " ", line).split()
        return reg_line

    def cleanLine(self, line):
        ''' 
        1. Removes punctuations
        2. lower case to all words

        Returns a clean string
        '''
        tokens = self.wordTokenize(line)
        words = [self.filterOutPunctuation(word).lower() for word in tokens]
        return " ".join(words)

    def clean_data(self, path):
        tr, dev, tst = [], [], []
        if self.dataset_type == 'fb':
            '''
            (<tokenized sent>, <top intent>)
            '''
            train = open(path+"train.tsv", 'r')
            test = open(path+"test.tsv", 'r')
            for line in train:
                curr_line = line.split("\t")
                cleaned_line = self.cleanLine(line[1])
                curr_y = fb_top_intent(curr_line[2])
                tr.append((cleaned_line, curr_y))
            for line in test:
                curr_line = line.split("\t")
                cleaned_line = self.cleanLine(line[1])
                curr_y = fb_top_intent(curr_line[2])
                tst.append((cleaned_line, curr_y)) 
            N = len(tr)
            tr, dev = tr[:(0.8*N)], tr[(0.8*N):]
        
        if self.dataset_type == 'atis':
            '''
            (<tokenized sent>, <top intent>)
            '''
            pass
            
        return tr, dev, tst


def split_data(train, dev, test):
    pass

def data_loader(path, dataset='fb'):
    dataCleaner = DataCleaner(tokenization_type=1, dataset=dataset)
    data_tr, data_dev, data_tst = dataCleaner.clean_data(path)
    
    return [], [], []
