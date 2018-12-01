# TF-IDF
Query rewrite in COM-AID model.
---
## Introduction
This project attempt to build a query-rewriter, which converts the raw queries into **clean and in-vocab** text. It is a preparation for feeding 
the COM-AID model.

## Algorithm
### 1. TF-IDF
IF-IDF is a comparatively light weighting schedule to find the near concepts of a specific given query.
However, it is based on a prerequisite that all the words in the text should be in the vocalbuary. To address this problem, we can replace the 
out-of-vocab words with in-vocab words using KNN.
### 2. Ball Tree and KNN
Brute force is too time-consuming in finding the top K nearest neighbors. We build a ball tree to organize the data set(word embeddings) instead of simple list.
A ***O(lgn)*** algorithm based on tree structure would be better than ***O(n)*** based on BF.

## Defect
If words in the query has no same words(totally different) with the corresponding concepts, TF-IDF can hard to find it in the candidates.

