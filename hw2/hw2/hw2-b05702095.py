#!/usr/bin/env python
# coding: utf-8

# In[103]:


import os
import sys
import nltk
import time
nltk.download('stopwords') # download stopwords lexion
nltk.download('punkt')     # download tokenize related tools 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from num2words import num2words
from collections import Counter

t1 = time.time()

# read stopwords
stopwordset = set( stopwords.words( 'english' ) )
stopwordset.update( { num2words( i ) for i in range( 0, 20 ) } ) # update english numbers to stopwords list  

# initialize porter stemmer
stemmer = PorterStemmer()
    
    
# function for tokenize -> remove stopwords -> stem
def Preprocessing( sentence ):
    
    # tokenize
    words = word_tokenize( sentence )
 
    # remove stopwords and stemming, return a list
    words = [ stemmer.stem( w.lower() ) for w in words if w.isalpha() and w.lower() not in stopwordset ] 
    
    return words

# use Counter to calculate df
Dictionary = Counter()

# read all the corpus
# os.walk is slow
for fname in os.listdir("./IRTMhw2Corpus/IRTM/"):
    
    # it is slow here, but I still trying to find out why
    with open( './IRTMhw2Corpus/IRTM/' + fname, 'r' ) as f:
        text = set( Preprocessing( f.read() ) )
        f.close()
        
    Dictionary.update( Counter( text ) )    
        

# sort the counter
ascending_dict = [ (i, Dictionary[i]) for i in sorted( Dictionary ) ]

# write to dictionary.txt
with open( 'dictionary.txt', 'w') as f:
    
    f.write('t_index term df\n')
    
    for i in range( len(ascending_dict) ):
        
        f.write( " ".join( [ str(i + 1) , ascending_dict[i][0], str(ascending_dict[i][1]) ] ) )
        if i != len(ascending_dict) - 1:
            f.write('\n')
    
    f.close()


# build dictionary
# word: (id, df)
with open( 'dictionary.txt', 'r' ) as f:
    
    text_dict = { w[1]: (w[0], int(w[2])) for w in [ t.split() for t in f.read().split('\n')[1:] ] }
    f.close()

print( 'Already built dictionary.txt!' )

os.mkdir('IRTMhw2tfidfVec')

# calculate idf

N = len( os.listdir('./IRTMhw2Corpus/IRTM/') )
def idf( word ):
    return np.log10( N / text_dict[word][1] )


# make tfidf file's string
# TFIDF 101

def MakeTfidfDoc( text ):

    doc = Preprocessing( text )
    term_id_list = []
    
    for w in set( doc ):

        # append( ( word_ID, tfidf ) )
        # do Sub-linear TF scaling
        
        tf_value = 1 + np.log( doc.count( w ) ) if doc.count( w ) > 0 else 0
        tfidf_value = tf_value * idf( w )
        term_id_list.append( (int( text_dict[w][0] ), tfidf_value ) )

    # sort word_ID to implement O(n) dot product in Cosine calculation
    term_id_list = sorted( term_id_list, key = lambda x: x[0] )
    tfidf_list = [ t[1] for t in term_id_list ]
    
    term_vec = np.array( tfidf_list )
    term_vec /= np.linalg.norm( term_vec )
    
    string = ''
    string += str( len(term_id_list) )
    string += '\nt_index tf-idf\n'
    
    for i in range( len( term_id_list ) ):
        string += str( term_id_list[i][0] ) + ' ' + str( term_vec[i] ) + '\n'
    
    return string[:-1]

# iterate over all .txt

for fname in os.listdir("./IRTMhw2Corpus/IRTM/"):
    with open( './IRTMhw2Corpus/IRTM/' + fname, 'r' ) as f:
        tfidfdoc = MakeTfidfDoc( f.read() )
        f.close()
            
    with open( './IRTMhw2tfidfVec/' + fname, 'w' ) as f:
        f.write( tfidfdoc )
        f.close()
            
    print( fname, 'has built tfidf vector file.' )
        
print( 'All docs have TFIDF form!' )


# Vectorize Read in TFIDF File's text
def Vectorize( string ):
    return [( int(t[0]), float(t[1]) ) for t in [ v.split() for v in string.split('\n')[2:] ] ]


# Calculate cosine between 2 vectors in O(n)
def Cosine( vec1, vec2 ):
    
    cosine = 0.0
    length = len(vec1) if len(vec1) <= len(vec2) else len(vec2)
    
    i1 = 0
    i2 = 0
    while i1 < length and i2 < length:
        
        if vec1[i1][0] == vec2[i2][0]:
            cosine += vec1[i1][1] * vec2[i2][1]
            i1 += 1
            i2 += 1
            
        elif vec1[i1][0] > vec2[i2][0]:
            i2 += 1
            
        else:
            i1 += 1
            
    return cosine


# calculate 1 & 2's cosine
with open('./IRTMhw2tfidfVec/1.txt', 'r') as f:
    vect1 = Vectorize(f.read())
    f.close()
    
with open('./IRTMhw2tfidfVec/2.txt', 'r') as f:
    vect2 = Vectorize(f.read())
    f.close()
    
print( 'Cosine between 1 & 2 tfidf-unit-vector is:', Cosine( vect1, vect2 ) )
# Cosine between 1 & 2 tfidf-unit-vector is: 0.15328033704552962

print( 'Running time is: ', time.time() - t1 )
