import os
import string

from nltk.tokenize import word_tokenize
import sklearn

class DataCleaner(object):

    def __init__(self, tokenization_type):
        self.token_type = tokenization_type 

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
        fil = open('r', path)
        lis = []
        for line in fil:
            #TODO: Add X, y differentiation
            cleaned_line = self.cleanLine(line)
            lis.append(cleaned_line)
        return lis

def split_data(train, dev, test):
    pass

def data_loader(path):
    dataCleaner = DataCleaner(tokenization_type=1)
    data = dataCleaner.clean_data(path)
    
    return [], [], []
