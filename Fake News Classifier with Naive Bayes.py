#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt


# In[2]:


#df =pd.read_csv('../input/fake-news/train.csv')

df =pd.read_csv(r'C:\Code\Problems\Fake News Classification\fake-news\train.csv')

# In[3]:


df.head()


# In[4]:


x= df.drop('label',axis=1)


# In[5]:


x.head(2)


# In[6]:


y = df['label']


# In[7]:


df.shape


# In[8]:


df.info()


# ### Check any Null Values in the dataframe

# In[9]:


df.isnull().sum()


# In[10]:


df=df.dropna()


# In[11]:


df.head()


# In[12]:


df['title'][3]


# In[13]:


messeges =df.copy()


# In[14]:


messeges.reset_index(inplace=True)


# In[15]:


messeges.head()


# In[16]:


corpus = []
for i in range(0, len(messeges)):
    review = re.sub('[^a-zA-Z]', ' ', messeges['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[17]:


corpus[6]


# ## Counter Vectorization
# ### Bag of Words

# In[18]:


cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()


# In[19]:


# show resulting vocabulary; the numbers are not counts, they are the position in the sparse vector.
cv.vocabulary_


# In[20]:


X.shape


# In[21]:


y=messeges['label']


# In[22]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[23]:


cv.get_feature_names()[:20]


# In[24]:


cv.get_params()


# In[25]:


count_df = pd.DataFrame(X_train, columns=cv.get_feature_names())


# In[26]:


count_df.head()


# In[27]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Mulinomial Naive Bayes Theorem

# In[29]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
from sklearn import metrics
import numpy as np
import itertools


# In[30]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[31]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[32]:


y_train.shape


# ### Multinomial Classifier with Hyperparameter

# In[39]:


classifier=MultinomialNB(alpha=0.1)


# In[40]:


previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))


# In[41]:


## Get Features names
feature_names = cv.get_feature_names()


# In[42]:


classifier.coef_[0]


# In[43]:


### Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]


# In[ ]:


### Most fake
sorted(zip(classifier.coef_[0], feature_names))[:5000]

