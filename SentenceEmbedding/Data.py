import os
import string
import re
import json
import logging

from nltk.tokenize import word_tokenize
import sklearn

from utils import fb_top_intent

logger = logging.getLogger(__name__)


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
            train = open(path + "train.tsv", 'r')
            test = open(path + "test.tsv", 'r')
            for line in train:
                curr_line = line.split("\t")
                cleaned_line = self.cleanLine(curr_line[1])
                curr_y = fb_top_intent(curr_line[2])
                tr.append((cleaned_line, curr_y))
            for line in test:
                curr_line = line.split("\t")
                cleaned_line = self.cleanLine(curr_line[1])
                curr_y = fb_top_intent(curr_line[2])
                tst.append((cleaned_line, curr_y))
            tr, dev = split_data(tr)

        if self.dataset_type == 'atis':
            '''
            (<tokenized sent>, <top intent>)
            '''
            tr = self.read_json(path + 'atistrain.json')
            dev = self.read_json(path + 'atisdev.json')
            tst = self.read_json(path + 'atistest.json')

        return tr, dev, tst

    def read_json(self, file_name):
        lst = []
        with open(file_name) as f:
            data = json.load(f)
            for elem in data['body']:
                curr_line = elem['text']
                cleaned_line = self.cleanLine(curr_line)
                curr_y = elem['intent']
                lst.append((cleaned_line, curr_y))
        return lst

class BERTInputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label=label


class BERTInputFeatures(object):
    
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def split_data(tr, split=0.8):
    N = len(tr)
    tr, dev = tr[:int(split * N)], tr[int(split * N):]
    return tr, dev

def data_loader(path, dataset='fb'):
    dataCleaner = DataCleaner(tokenization_type=1, dataset=dataset)
    data_tr, data_dev, data_tst = dataCleaner.clean_data(path)
    return data_tr, data_dev, data_tst

def create_bert_examples(lines, set_type):
    examples = []
    for i, line in enumerate(lines):
        examples.append(
            BERTInputExample(
                guid='%s-%s' % (set_type, i),
                text_a=line[0],
                label=line[1]
            )
        )
    return examples

def bert_examples_to_features(examples, label_list, max_seq_len, tokenizer):
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for ex_idx, ex in enumerate(examples):
        tokens_a = tokenizer.tokenize(ex.text_a)

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = token_a[:(max_seq_len - 2)]

        tokens = ['[CLS]'] + tokens_a + ['[SEP]']

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        label_id = label_map[ex.label]

        if ex_idx < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (ex.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (ex.label, label_id))

        features.append(
            BERTInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id
            )
        )

    return features
