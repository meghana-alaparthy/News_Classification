# -*- coding: utf-8 -*-
"""FinalCode.ipynb

**Import Libraries**
"""

import nltk
import copy
import re
import numpy
import matplotlib.pyplot as pypt
nltk.download('stopwords')
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.neural_network import MLPClassifier

"""**Prepare Data**"""

titles = []
categories = []
with open('dsjVoxArticles.tsv','r') as tsv:
    count = 0;
    for line in tsv:
        x = line.strip().split('\t')[:3]
        if x[2] in ['Politics & Policy', 'Health Care', 'Business & Finance', 'Criminal Justice', 'Science & Health']:
            title = x[0].lower()
            title = re.sub('\s\W',' ',title)
            title = re.sub('\W\s',' ',title)
            titles.append(title)
            categories.append(x[2])

"""**Split Data**"""

title_tr, title_te, category_tr, category_te = train_test_split(titles,categories)
title_tr, title_de, category_tr, category_de = train_test_split(title_tr,category_tr)
print("Training : ",len(title_tr))
print("Developement : ",len(title_de),)
print("Testing : ",len(title_te))

"""**Vectorization of data**"""

tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)
vectorizer.fit(iter(title_tr))
Xtr = vectorizer.transform(iter(title_tr))
Xde = vectorizer.transform(iter(title_de))
Xte = vectorizer.transform(iter(title_te))
encoder = LabelEncoder()
encoder.fit(category_tr)
Ytr = encoder.transform(category_tr)
Yde = encoder.transform(category_de)
Yte = encoder.transform(category_te)

"""**Feature Reduction**"""

print("No. of features before reduction : ", Xtr.shape[1])
selection = VarianceThreshold(threshold=0.001)
Xtr_whole = copy.deepcopy(Xtr)
Ytr_whole = copy.deepcopy(Ytr)
selection.fit(Xtr)
Xtr = selection.transform(Xtr)
Xde = selection.transform(Xde)
Xte = selection.transform(Xte)
print("No. of features after reduction : ", Xtr.shape[1])

"""**Data Sampling**"""

labels = list(set(Ytr))
counts = []
for label in labels:
    counts.append(numpy.count_nonzero(Ytr == label))
pypt.pie(counts, labels=labels, autopct='%1.1f%%')
pypt.show()

s = SMOTE(random_state=42)
Xtr, Ytr = s.fit_sample(Xtr, Ytr)
labels = list(set(Ytr))
counts = []
for label in labels:
    counts.append(numpy.count_nonzero(Ytr == label))
pypt.pie(counts, labels=labels, autopct='%1.1f%%')
pypt.show()

"""**Training Models**

**Baseline Model**
"""

dc = DummyClassifier(strategy="stratified")
dc.fit(Xtr, Ytr)
p = dc.predict(Xde)
print(classification_report(Yde, p, target_names=encoder.classes_))

"""**Random Forest**"""

rf = RandomForestClassifier(n_estimators=50)
rf.fit(Xtr, Ytr)
p = rf.predict(Xde)
print(classification_report(Yde, p, target_names=encoder.classes_))

"""**Ada Boost Classifier**"""

abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=50)
abc.fit(Xtr,Ytr)
p = abc.predict(Xde)
print(classification_report(Yde,p,target_names=encoder.classes_))

"""**Gradient Boost Classifier**"""

gbc = GradientBoostingClassifier(n_estimators=50,learning_rate=0.5,max_features=2,max_depth=2,random_state=0)
gbc.fit(Xtr,Ytr)
p=gbc.predict(Xde)
print(classification_report(Yde,p,target_names=encoder.classes_))

"""**Multinomial Naive Bayes**"""

mnb = MultinomialNB()
mnb.fit(Xtr, Ytr)
p = mnb.predict(Xde)
print(classification_report(Yde, p, target_names=encoder.classes_))

"""**Support Vector Classification**"""

svc = SVC()
svc.fit(Xtr, Ytr)
p = svc.predict(Xde)
print(classification_report(Yde, p, target_names=encoder.classes_))

"""**Multilayer Perceptron**"""

mp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1, max_iter=400)
mp.fit(Xtr, Ytr)
p = mp.predict(Xde)
print(classification_report(Yde, p, target_names=encoder.classes_))

"""**Comparing the results, the best model is Multinomial Naive Bayes**

**Testing**
"""

p = mnb.predict(Xte)
print(classification_report(Yte, p, target_names=encoder.classes_))
sns.heatmap(confusion_matrix(Yte, p))

dupnb = MultinomialNB()
dupnb.fit(Xtr_whole, Ytr_whole)
coefs = dupnb.coef_
target_names = encoder.classes_

reverse_vocabulary = {}
vocabulary = vectorizer.vocabulary_
for word in vocabulary:
    index = vocabulary[word]
    reverse_vocabulary[index] = word

for i in range(len(target_names)):
    words = []
    for j in coefs[i].argsort()[-20:]:
        words.append(reverse_vocabulary[j])
    print (target_names[i], '-', words, "\n")