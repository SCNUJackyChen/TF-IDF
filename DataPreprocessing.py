import nltk
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import xlwt

pd_concepts = pd.read_csv('data.csv',encoding='ISO-8859-1')
np_concepts = np.array(pd_concepts)
np_concepts_list = np_concepts.tolist() # shape --> [ [str],[str],...,[str] ]
corpus_raw = [''.join(s[0]) for s in np_concepts_list] # shape --> [str,str,...,str]

def get_tokens(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens # shape --> [str,str,...,str]

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def stemming(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    return stemmed # shape --> [str,str,...str,]

def list2str(lis):
    s = ''
    for word in lis[0:-1]:
        s = s + word + ' '
    s = s + lis[-1]
    return s

def preprocessor():
    corpus = []
    count = 0
    wb = xlwt.Workbook()
    table = wb.add_sheet('Sheet1')
    for concept_raw in corpus_raw:
        concept = stemming(concept_raw)
        if len(concept) == 0:
            continue
        concept = list2str(concept)
        table.write(count,0,concept)
        # corpus.append(concept)

        print("finished",count)
        count += 1
    wb.save('cleaned_data.xls')
    return corpus # shape --> [ [str], [str], ..., [str] ]


# =====================test=============================
# text1 = 'Tuberculous abscess of brain, bacteriological or histological examination unknown (at present)'
# count = Counter(stemmed)
# print(count)
# print(stemming(text1))
# preprocessor()
# a = []
# print([list2str(a)])