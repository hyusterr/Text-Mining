import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from collections import Counter
import operator
from random import randint
import pickle
import re
import os
from sklearn.feature_selection import chi2
import random

def readPickle():
	with open('dataset.pickle', 'rb') as data:
		data = pickle.load(data)
	return(data)

def removeNode(data):
	r1 = '[0-9’!１２３４５６７８９０"#$%&\'()（）*+,-/:;<=>?@，。?★、…【】《》＊;「」？“”‘’！：[\\]^_`{|}~‧，？.～ ；]+'
	for i in range(len(data)):
	#for i in range(1):
		for j in range(len(data[i][1])):
			data[i][1][j] = re.sub(r1,'',data[i][1][j])

	for i in range(len(data)):
		needRemove = 0
		for j in range(len(data[i][1])):
			if data[i][1][j-needRemove] == '':
				#print(j,j-needRemove,)
				data[i][1].remove(data[i][1][j-needRemove])
				needRemove += 1
	return(data)

def loadStopwords():
	stopwords = []
	with open('CN_stopwords.txt', 'r', encoding='UTF-8') as f1:
		for line in f1:
			stopwords += (line.split())
	return(stopwords)

def splitTrainTest(data):
	len_all = len(data)
	boundary = int(len_all*7/8)
	print(boundary)
	random.shuffle(data)
	train_data = data[0:boundary]
	test_data = data[boundary:]
	train_size = boundary
	test_size = len_all - boundary


	return(train_data, test_data, train_size, test_size)

def makeDataList(train_data, test_data):
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	train_str = []
	test_str = []
	for i in range(len(train_data)):
		train_x.append(train_data[i][1])
		train_y.append(train_data[i][0])
	for i in range(len(test_data)):
		test_x.append(test_data[i][1])
		test_y.append(test_data[i][0])

	for i in range(len(train_x)):
		str1 = ' '.join(train_x[i])
		str1 = re.sub('\n','',str1)
		train_str.append(str1)

	for i in range(len(test_x)):
		str1 = ' '.join(test_x[i])
		str1 = re.sub('\n','',str1)
		test_str.append(str1)
	return(train_str, test_str, train_y, test_y)

def dataRepresent(CN_stopwords, train_str, test_str):
	tfidfconverter = TfidfVectorizer(max_features = feature_num, min_df=5, max_df=0.6, stop_words=CN_stopwords) 
	print("======== method : tfidfconverter =========")
	train_x = tfidfconverter.fit_transform(train_str).toarray()
	features = tfidfconverter.get_feature_names()
	print("======== features : tfidfconverter =========")
	print(tfidfconverter.get_feature_names())
	temp = TfidfVectorizer(vocabulary=features)
	test_x = temp.fit_transform(test_str).toarray()

	return(train_x, test_x)

# =========================== main function ===========================
feature_num = 420
data = readPickle()
data = removeNode(data)
CN_stopwords = loadStopwords()
train_data, test_data, train_size, test_size= splitTrainTest(data)
train_str, test_str, train_y, test_y = makeDataList(train_data, test_data)
train_x, test_x = dataRepresent(CN_stopwords, train_str, test_str)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(train_x, train_y[:train_size])

cnt = 0
for i in range(test_size):
    if np.sum(test_x) == 0:
        cnt = cnt+1
cntB = 0
for i in range(train_size):
    if np.sum(train_x[i]) == 0:
        cntB = cntB+1

#%%
pred_y = classifier.predict(test_x) 

#%%
#print(len(y_pred))
#%%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_y,pred_y ))  
print(classification_report(test_y,pred_y))  
print(accuracy_score(test_y, pred_y))
print("=============== cnt =================", "\n", "test = ", cnt, "\n", "train = ", cntB)

#print(data)
# print(data[0][1])
#用feature selection做model。
with open('model.pickle', 'wb') as f:
    pickle.dump(classifier, f)
    f.close()