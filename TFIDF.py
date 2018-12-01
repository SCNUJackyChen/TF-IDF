import DataPreprocessing
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import nltk
import xlwt
import NN
import re
from pprint import pprint

#  ========================== Data Preparation ==============================
a = pd.read_csv('cleaned_data.csv', encoding='ISO-8859-1')
b = np.array(a)
c = b.tolist()
corpus = []
for concept in c:
    d = DataPreprocessing.list2str(concept)
    corpus.append(d)

a = pd.read_csv('concept_dict.csv', encoding='ISO-8859-1')
b = np.array(a)
c = b.tolist()
dictionary = []
for word in c:
    dictionary.append(word[0])


def query_rewrite(text: str, k: int):

    # First, rewrite each word in the query text. Replace the out-of-vocab words with
    # their nearest words in embedding using KNN of ball-tree.
    text_list = re.split('[\\\\+\-#/,|;-?*$%()\[\]\s]', text)  #  clean data firstly
    text_list = [NN.word_rewrite(w) for w in text_list]
    text = DataPreprocessing.list2str(text_list)
    text = DataPreprocessing.list2str(DataPreprocessing.stemming(text))
    print(text)
    #  Second, add this query to our corpus and calculate the TF-IDF value

    corpus.append(text)
    corpus_array = np.array(corpus)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    #  Third, calculate the cosine closest concepts of the query. Return the top K concepts.
    cosine_similarities = linear_kernel(tfidf[-1:], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-k-1:-1]

    return corpus_array[related_docs_indices]
    # print(cosine_similarities[related_docs_indices])


# ======================== test =====================================
#  Experiment:
# 'aortic stenosis\cardiac cath' --> top 40
# 'chest pain\cath'  --> top 60+
# 'newborn'  --> top 60+

text1 = 'chest pain\cath'
if __name__ == '__main__':
    result = query_rewrite(text1, 60)
    pprint(result)