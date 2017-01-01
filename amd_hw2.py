# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:12:33 2016

@author: Federico
"""

import pandas as pd
import numpy as np
import math
import os
import nltk
from nltk.stem.snowball import EnglishStemmer
import re
import sys
from sklearn.metrics.pairwise import cosine_similarity

reload(sys)
sys.setdefaultencoding('utf-8')

def storIndex(dic):
    '''Store the vocabulary and postings list'''
    keys = sorted(dic.keys())
    k=0
    with open("vocabulary.tsv", "w+") as f1:
        with open("postings.tsv", "w+") as f2:
            for key in keys:
                f1.write(str(k) +"\t"+ key+"\n")
                f2.write(str(k) + "\t")
                for doc in sorted(dic[key]): 
                    f2.write(str(doc) + "\t")
                f2.write("\n")
                k=k+1

def retIndex():
    '''Build the inverted index from vocabulary and postings saves'''
    index = {}
    with open("vocabulary.tsv", "r") as v:
        with open("postings.tsv", "r") as p:
            for linev, linep in zip(v, p):
                linev = linev.split()
                linep = linep.split()
                index[linev[1]] = linep[1:]
    return index
    
    
def numericalSort(value):
    '''Utils to sort file names'''
    
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def index():
    '''Build the inverted index from the stored recipes'''
    dic = {}
    
    # Folder of the stores recipes
    docs = sorted(os.listdir("doc-hw/"), key=numericalSort)
    
    for i,doc in enumerate(docs):
        
        try:
            #Read tab separated file of the recipe
            df = pd.read_csv("doc-hw/"+str(i)+".tsv", sep="\t", encoding="utf-8")
        except IOError:
            #Debug
            print str(i)
            continue
        rec = ''
        
        for column in df:
            if column == 'Unnamed: 0' or column == 'link':
                continue
            else:
                # Build a string composed from the given attributes
                rec += str(df[column][0]) + ' '
        
        #Find all the words
        rec = re.findall(r"[\w']+" , rec)
        
        for word in rec:
            # Stemmer
            word = EnglishStemmer().stem(word).lower()
            # Stopowords
            if not word in nltk.corpus.stopwords.words("english"):
                if not word in dic:
                    # If word is not in dictionary create a key,doc
                    dic[word] = {i}
                else:
                    # Add document to the word entry in the index
                    dic[word].add(i)
                    
    # store the index
    storIndex(dic)
    return dic


def cleanQuery(query, index):
    '''Utils to clean a query'''
    
    cleanedQuery = []
    cleanedWords = []
    for word in query:
        if word in index:
            cleanedQuery.append(word)
        else:
            cleanedWords.append(word)
    return cleanedQuery, cleanedWords
    
def same(items):
    '''Utils to check all items are the same'''
    return all(x == items[0] for x in items)
    
def search(query, index):
    '''Search function'''
    
    index = index
    # Split the query in input
    query = query.split(" ")
    # Stem the query
    query = [EnglishStemmer().stem(word).lower() for word in query]
    # Clean the query (check if all words are in the index)
    query, cleanedWords = cleanQuery(query, index)
    
    # If no word is in the inverted index we can't perform the search, ret 0
    if len(query) == 0:
        return 0
    
    
    ###### Skip Pointers Implementation
    
    pointers = {}
    
    # Pointers for all the words to be searched
    for word in query:
        pointers[word] = 0
        
    
    resp = []
    min_ = query[0]
    
    
    finished = False
    
    # Begin the search with the pointers
    while not finished:
        
        buff = [int(index[word][pointers[word]]) for word in query]
        
        if same(buff):
            resp.append(buff[0])
            
        aux = sorted(buff)
        min_ = aux[0]
        max_ = aux[-1]
        
        for word in query:
            
            # For all the word in query start the pointers
            if int(index[word][pointers[word]]) == min_:
                
                # Pointers move with 4 then return at the start if pointer is less
                if pointers[word]+4 < len(index[word]) -1 and int(index[word][pointers[word]+4]) < max_:
                    pointers[word] = pointers[word] + 4
                    
                elif pointers[word]+1 < len(index[word]) -1:
                    pointers[word] = pointers[word] + 1
                    
                else:
                    finished = True
    # No reponse from the search return 1 (handled in main)            
    if len(resp) == 0:
        return 1
    
    return resp, query, cleanedWords
    
def createBoW(ind, iTw, wTi):
    '''Create BoW/Tfidf vectors for every recipe'''
    
    # List of recipes
    docs = sorted(os.listdir("doc-hw/"), key=numericalSort)
    
    list_of_doc = []
    
    # How many words we have
    len_vocab = len(ind.keys())
        
    for i,doc in enumerate(docs):
        # Initialize array with the given length 
        bow_array = np.zeros(len_vocab)
        
        try:
            # Open doc
            df = pd.read_csv("doc-hw/"+str(i)+".tsv", sep="\t", encoding="utf-8")
        except IOError:
            print str(i)
            continue
            
        rec = ''
        
        #Create a string
        for column in df:
            if column == 'Unnamed: 0' or column == 'link':
                continue
            else:
                rec += str(df[column][0]) + ' '
        
        # Find the words
        rec = re.findall(r"[\w']+" , rec)
        
        # Stem the word and add occurrency in document
        for i,word in enumerate(rec):
            word = EnglishStemmer().stem(word).lower()
            if not word in nltk.corpus.stopwords.words("english"):
                bow_array[int(wTi[word])] += 1
        
        # Compute tfidf for every word in vector
        for i,elem in enumerate(bow_array):
            bow_array[i] = bow_array[i] * (1 + math.log(len_vocab/(len(ind[iTw[str(i)]]))))
        
        # Append final vector to list of tfidf vectors (l2 normalization)
        list_of_doc.append(bow_array/np.linalg.norm(bow_array))
    return list_of_doc
    
def transformQuery(query, ind, map_ind_word, map_word_ind):
    '''Transform the query in a tfidf form'''
    bow_query = np.zeros(len(ind))
    
    # Frequency
    for i,word in enumerate(query):
        if not word in nltk.corpus.stopwords.words("english"):
            bow_query[int(map_word_ind[word])] += 1
    
    # Idf
    for i,elem in enumerate(bow_query):
        bow_query[i] = bow_query[i] * (1 + math.log(len(ind)/(len(ind[map_ind_word[str(i)]]))))
    
    # Return l2 normalized
    return bow_query/np.linalg.norm(bow_query)

def createMappings():
    '''Utils to store dictionaries of index to word and word to index mappings'''
    with open("vocabulary.tsv", "r") as v:
        word_to_ind = {}
        ind_to_word = {}

        for linev in v:
            linev = linev.split()
            word_to_ind[linev[1]] = linev[0]
            ind_to_word[linev[0]] = linev[1]
            
    return ind_to_word, word_to_ind
    
def ranking(tfidf_query, list_tfidf, res):
    '''Compute the cosine similarity and return best matches'''
    arr_for_sim = []
    for elem in res:
        arr_for_sim.append([elem,list_tfidf[elem]])
    
    # Cosine between query and list of returned vectors
    for i,elem in enumerate(arr_for_sim):
        arr_for_sim[i][1] = cosine_similarity(tfidf_query.reshape(1,-1), arr_for_sim[i][1].reshape(1,-1))
    
    #Sort for most relevant
    arr_for_sim.sort(key=lambda x: -x[1])
    
    return [i[0] for i in arr_for_sim]
    
def main():
    
    # Create the inverted index and store it
    index()
    # read inverted index from stored file
    inverted_index = retIndex()
    # Mapping index to word (vocab) and word to index
    ind_to_word, word_to_ind = createMappings()
    # Create tfidf vectors from recipes
    list_of_tfidf = createBoW(inverted_index, ind_to_word, word_to_ind)
    
    
    
    
    while True:
        
        q = str(raw_input("Please enter your search: "))
        # Search the query in index
        result = search(q,inverted_index)
        
        if result == 0:
            print "No result has been found for the given query"
            continue
        
        if result == 1:
            print "No recipe has simultaneously all the words in the query"
            continue
        
        trasformed_query = transformQuery(result[1], inverted_index, ind_to_word, word_to_ind)
        
        # Return best 10
        rank = ranking(trasformed_query, list_of_tfidf, result[0])[:10]
        
        for elem in rank:
            df = pd.read_csv("doc-hw/"+str(elem)+".tsv", sep="\t", encoding="utf-8")
            print df['link'] + '\n'
            
        
        
    
    
    print "done"

if __name__ == '__main__':
    main()
