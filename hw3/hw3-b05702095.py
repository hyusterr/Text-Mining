#!/usr/bin/env python
# coding: utf-8

# In[46]:


import os
import sys
import nltk
nltk.download('stopwords') # download stopwords lexion
nltk.download('punkt')     # download tokenize related tools 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from num2words import num2words
from collections import Counter


# In[23]:


# load data
with open('./training.txt', 'r') as f:
    training = f.readlines()
    f.close()


# In[24]:


# prepare training data
classes = [i.strip().split()[0]  for i in training]
docs    = [i.strip().split()[1:] for i in training]
class_N = np.array([len(i) for i in docs])

training_N = 0
for cls in docs:
    training_N += len(cls)
    
prior = class_N / training_N


# In[95]:


# read stopwords
stopwordset = set( stopwords.words( 'english' ) )
# update english numbers to stopwords list  
stopwordset.update( { num2words( i ) for i in range( 0, 20 ) } ) 

# initialize porter stemmer
stemmer = PorterStemmer()

    
# function for tokenize -> remove stopwords -> stem
def Preprocessing( sentence ):
    
    # tokenize
    words = word_tokenize( sentence )
 
    # remove stopwords and stemming, return a list
    words = [ stemmer.stem( w.lower() ) for w in words if w.isalpha() and w.lower() not in stopwordset ] 
    
    # return a list
    return words


def TrainMultiNB( docs ):
    # get the dimension of all words
    Dictionary = set()

    # read through training corpus
    for cls in docs:
    
        for doc in cls:
    
            with open( 'IRTM/' + doc + '.txt', 'r' ) as f:
                text = set( Preprocessing( f.read() ) )
                f.close()
        
            Dictionary.update( text )  
        

    # sort the counter
    Dictionary = sorted(list(Dictionary))

    print( 'Already built MutliNomialNB Dictionary!' )
    print('length of dictionary: ', len(Dictionary))

    # build word position dictionary
    pos_dict = { Dictionary[i]: i for i in range(len(Dictionary)) }
    term_N   = len(Dictionary)

    # build class-TF matrix
    cls_tf_matrix = []

    for cls in docs:
    
        cls_tf_vec = [1] * term_N
    
        for doc in cls:
        
            with open('IRTM/' + doc + '.txt', 'r') as f:
                text = Preprocessing( f.read() )
                f.close()
            
            for term in text:
                cls_tf_vec[ pos_dict[term] ] += 1
            
        cls_tf_matrix.append( cls_tf_vec )
    
    
    cls_posterior_matrix = np.array( [ np.array(vec) / sum(vec) for vec in cls_tf_matrix ] )

    print( 'the MultiNomial NB model is built, the shape is:', cls_posterior_matrix.shape )
    
    # return probability matrix, term frequency matrix, feature position matrix
    
    return cls_posterior_matrix, np.array( cls_tf_matrix ), pos_dict


# In[136]:


def NBPredict( cls_posterior_matrix, pos_dict, text ):

    classes = [ str(i) for i in range(1, 14) ]
    
    doc = Preprocessing( text )
    cls_scores = []
    
    for cls in range(len(prior)):
        
        cls_score = np.log(prior[cls])
        
        for term in doc:
            if term in pos_dict:
                cls_score += cls_posterior_matrix[cls][pos_dict[term]]
                
        cls_scores.append(cls_score)
        
    return classes[ np.argmax( np.array(cls_scores) ) ]


# In[109]:


def TrainBernoulliNB( docs ):
    # get the dimension of all words
    Dictionary = set()

    # read through training corpus
    for cls in docs:
    
        for doc in cls:
    
            with open( 'IRTM/' + doc + '.txt', 'r' ) as f:
                text = set( Preprocessing( f.read() ) )
                f.close()
        
            Dictionary.update( text )  
        

    # sort the counter
    Dictionary = sorted(list(Dictionary))

    print( 'Already built Bernoulli Dictionary!' )
    print('length of dictionary: ', len(Dictionary))

    # build word position dictionary
    pos_dict = { Dictionary[i]: i for i in range(len(Dictionary)) }
    term_N   = len(Dictionary)

    # build class-TF matrix
    cls_tf_matrix = []

    for cls in docs:
    
        cls_tf_vec = [1] * term_N
    
        for doc in cls:
        
            with open('IRTM/' + doc + '.txt', 'r') as f:
                text = set( Preprocessing( f.read() ) )
                f.close()
            
            for term in text:
                cls_tf_vec[ pos_dict[term] ] += 1
            
        cls_tf_matrix.append( cls_tf_vec )
    
    
    cls_posterior_matrix = np.array( [ np.array(vec) / sum(vec) for vec in cls_tf_matrix ] )

    print( 'the Bernoulli NB model is built, the shape is:', cls_posterior_matrix.shape )
    
    return cls_posterior_matrix, np.array( cls_tf_matrix ), pos_dict


# In[97]:


# prepare testing set

training_set = []
for i in docs:
    training_set += i

print( 'getting testing data' )
    
testing_set = []
for i in range(1, 1096):
    if str(i) not in training_set:
        testing_set.append(str(i))


# In[103]:


# training

print('Training MultiNomial NB...')
cls_condi_matrix, cls_tf_matrix, pos_dict = TrainMultiNB( docs )


# In[99]:


# predict on testing set

print('Predicting...')
out = []

for txt in testing_set:
    with open('./IRTM/' + txt + '.txt') as f:
        t = f.read()
        f.close()
    out.append(NBPredict( cls_condi_matrix, pos_dict, t ))


# In[100]:


outcsv = 'id,Value\n'

for idx, val in zip( testing_set, out ):
    outcsv += idx + ',' + val + '\n'

with open('MultiNB-out.csv', 'w') as f:
    f.write(outcsv)
    f.close()

print('MultiNB predicting is done! output is MultiNB-out.csv') 


# In[110]:


print('Training Bernoulli NB...')

# Bernoulli Training

ber_cls_prob_matrix, ber_cls_df_matrix, ber_pos_dict = TrainBernoulliNB( docs )


# In[117]:

print('Predicting...')

# predict on testing set

ber_out = []

for txt in testing_set:
    with open('./IRTM/' + txt + '.txt') as f:
        t = f.read()
        f.close()
    ber_out.append(NBPredict( ber_cls_prob_matrix, ber_pos_dict, t ))
    
ber_outcsv = 'id,Value\n'

for idx, val in zip( testing_set, ber_out ):
    ber_outcsv += idx + ',' + val + '\n'

with open('BerNB_out.csv', 'w') as f:
    f.write(ber_outcsv)
    f.close()


print('Bernoulli NB predicting is done! output is BerNB_out.csv')

# In[124]:


def CalculateChiScore( tf_vec ):
    
    chi_score = 0
    
    for cls in range( len( tf_vec ) ): # 13 classes
        
        present_ontopic  = tf_vec[cls]
        absent_ontopic   = len(docs[cls]) - present_ontopic
        present_offtopic = sum( tf_vec ) - present_ontopic
        absent_offtopic  = len( training_set ) - present_ontopic - present_offtopic - absent_ontopic
        
        present = present_offtopic + present_ontopic
        ontopic = present_ontopic  + absent_ontopic
        
        Ne = len( training_set ) * present / len( training_set ) * ontopic / len( training_set )
        
        chi_score += ( present_ontopic - Ne ) ** 2 / Ne
        
    return chi_score

# print( len(ber_cls_df_matrix[0,]))

print('Applying Chi-Score Feature Selection...')

chi_score_list = []

for i in range( len(ber_cls_df_matrix[0]) ):
    
    # return to original tf matrix
    # vector shape (13, 1)
    ori_df_vec = ber_cls_df_matrix[:,i] - 1 
#     print( ori_df_vec )
    chi_score_list.append( (i, CalculateChiScore(ori_df_vec) ) )
#     chi
#     absent   = len(training_set) - present
    


# In[130]:


chi_select = sorted( chi_score_list, key= lambda x: x[1], reverse=True )[:500]
pos_chi_list = [ x[0] for x in chi_select ]

print('Get top 500 important features!')

# In[131]:


print('Predicting on model after feature selection')

def ChiNBPredict( cls_posterior_matrix, pos_dict, pos_chi_list, text ):

    doc = Preprocessing( text )
    cls_scores = []
    
    for cls in range(len(prior)):
        
        cls_score = np.log(prior[cls])
        
        for term in doc:
            if term in pos_dict and pos_dict[term] in pos_chi_list:
                cls_score += cls_posterior_matrix[cls][pos_dict[term]]
                
        cls_scores.append(cls_score)
        
    return classes[ np.argmax( np.array(cls_scores) ) ]


# In[133]:


# predict on testing set

chi_out = []

for txt in testing_set:
    with open('./IRTM/' + txt + '.txt') as f:
        t = f.read()
        f.close()
    chi_out.append( ChiNBPredict( cls_condi_matrix, pos_dict, pos_chi_list, t) )
    
chi_outcsv = 'id,Value\n'

for idx, val in zip( testing_set, chi_out ):
    chi_outcsv += idx + ',' + val + '\n'

with open('chi-mul-out.csv', 'w') as f:
    f.write(chi_outcsv)
    f.close()


# In[135]:


# predict on testing set

chi_ber_out = []

for txt in testing_set:
    with open('./IRTM/' + txt + '.txt') as f:
        t = f.read()
        f.close()
    chi_ber_out.append( ChiNBPredict( ber_cls_prob_matrix, ber_pos_dict, pos_chi_list, t) )
    
chi_ber_outcsv = 'id,Value\n'

for idx, val in zip( testing_set, chi_ber_out ):
    chi_ber_outcsv += idx + ',' + val + '\n'

with open('chi-ber-out.csv', 'w') as f:
    f.write(chi_ber_outcsv)
    f.close()

print( 'Prediction is done! Files are chi-* files!' )

