#!/usr/bin/env python
# coding: utf-8

# # Reddit Subreddit Classification
# ## Notebook 3 - Preprocessing and Modeling
# ---

# ## Imports

# ### Libraries

# In[64]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datetime import datetime
pd.options.display.max_colwidth = 200


# ### Data

# In[46]:


final_df = pd.read_csv("./data/final.csv")


# ## Preprocessing
# Based on my EDA, I thought that it was worth trying two different attempts at changing the text data prior to modeling.
# 1. Consolidating words covering certain topics into one word
# 2. Trying additional stop words

# ### Consolidating Topic Words

# To implement the replacement of all words covering certain topics, I created a function that would run regexs composed to capture tokens covering those topics and replacing them with a single word for that topic.

# In[83]:


# Function to convert all tokens covering a topic into one word
def consolidate_topics(X):
    print("Running consolidate topics")
    # Regexs for topic words
    girl = r"girl[A-Za-z]*(?![0-9])"
    boy_guy = r"boy|guy"
    marriage = r"marr[A-Za-z]*"
    divorce = r"divo[A-Za-z]*"
    over_forty = r"\A[4-7][0-9]\Z|\A[4-7][0-9](?!pm|min)[^0-9k]"
    under_forty = r"\A[2-3][0-9]\Z|\A[2-3][0-9](?!pm|min)[^0-9k]"
    kids = r"\Akid[s]?\Z|kiddo|child|\Ason[s]?\Z|daughter|grandkid|grandchild|grandson"
    school = r"\Aschool|college|universi"

    # Tokenize per sklearn's regex sequence
    X = X.str.findall(r"(?u)\b\w\w+\b").apply(lambda x: " ".join(x))

    # Replace all different variations of topic with one word
    X = X.str.replace(girl, "girl", regex=True)
    X = X.str.replace(boy_guy, "boy", regex=True)
    X = X.str.replace(marriage, "marriage", regex=True)
    X = X.str.replace(divorce, "divorce", regex=True)
    X = X.str.replace(over_forty, "over_forty", regex=True)
    X = X.str.replace(under_forty, "under_forty", regex=True)
    X = X.str.replace(kids, "child", regex=True)
    X = X.str.replace(school, "school", regex=True)    
    return X


# In[84]:


# Create FunctionTransformer with the consolidate_topics function
consolidate_tf = FunctionTransformer(consolidate_topics, validate=False)


# ### Additional Stop Words
# The stop words below were found during EDA and will be tried out during model gridsearches.

# In[85]:


# Common words found during EDA that may work as stop words
add_stop_words = ["just", "like", "date", "dating", "ve", "want", 
                  "time", "relationship", "think", "feel", "people", 
                  "don", "really", "know", "said", "didn", "things",
                 "going", "good", "person"]

# Adding additional stop words to default stop words in sklearn's text vectorizer
new_stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)


# ## Modeling

# ### Declare X and y

# In[86]:


X = final_df["alltext"]
y = final_df["||__target__||"]


# ### Train test split

# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


# ### GridSearch Function

# In[89]:


def run_gridsearch(X_train, X_test, y_train, y_test, parameters):
    pipeline_items = []
    if parameters["consolidate_text"]:
        pipeline_items.append(("consolidate", consolidate_tf))
    pipeline_items.append(("vec", parameters['vec']))
    pipeline_items.append(("estimator", parameters['estimator']))
    
    pipe = Pipeline(pipeline_items)
    
    gridsearch_params = parameters['params']
    
    # Instantiate GridSearchCV
    gs = GridSearchCV(estimator=pipe, param_grid=gridsearch_params, n_jobs=-1, verbose = 4)

    # Fit GridSearch
    gs.fit(X_train, y_train)

    # Output results
    pd.DataFrame(gs.cv_results_).to_csv(f"./gridsearch_results/{parameters['name']}-{datetime.now()}.csv", index=False)
    
    # Print scores
    print(f"Train Score: {gs.score(X_train, y_train)}")
    print(f"Test Score: {gs.score(X_test, y_test)}")


# ### Logistic Regression Models

# In[90]:


log_reg_con_tf_params = {
    "name": "logistic_regression_consolidate_tfidf",
    "consolidate_text": True,
    "vec": TfidfVectorizer(),
    "estimator": LogisticRegression(solver="liblinear"),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__penalty": ["none", "l1", "l2", "elasticnet"],
        "estimator__C": [0.01,0.1,1,10]
    }
}


# In[ ]:


log_reg_con_count_params = {
    "name": "logistic_regression_consolidate_count",
    "consolidate_text": True,
    "vec": CountVectorizer(),
    "estimator": LogisticRegression(solver="liblinear"),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__penalty": ["none", "l1", "l2", "elasticnet"],
        "estimator__C": [0.01,0.1,1,10]
    }
}


# In[ ]:


log_reg_tf_params = {
    "name": "logistic_regression_tfidf",
    "consolidate_text": False,
    "vec": TfidfVectorizer(),
    "estimator": LogisticRegression(solver="liblinear"),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__penalty": ["none", "l1", "l2", "elasticnet"],
        "estimator__C": [0.01,0.1,1,10]
    }
}


# In[ ]:


log_reg_count_params = {
    "name": "logistic_regression_count",
    "consolidate_text": False,
    "vec": CountVectorizer(),
    "estimator": LogisticRegression(solver="liblinear"),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__penalty": ["none", "l1", "l2", "elasticnet"],
        "estimator__C": [0.01,0.1,1,10]
    }
}


# In[ ]:





# In[ ]:





# ### Multinomial Naive Bayes Models

# In[ ]:


mnb_con_tf_params = {
    "name": "mnb_consolidate_tfidf",
    "consolidate_text": True,
    "vec": TfidfVectorizer(),
    "estimator": MultinomialNB(),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__alpha": [0, 0.1, 0.5, 1]
    }
}


# In[ ]:


mnb_con_count_params = {
    "name": "mnb_consolidate_count",
    "consolidate_text": True,
    "vec": CountVectorizer(),
    "estimator": MultinomialNB(),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__alpha": [0, 0.1, 0.5, 1]
    }
}


# In[ ]:


mnb_tf_params = {
    "name": "mnb_tfidf",
    "consolidate_text": False,
    "vec": TfidfVectorizer(),
    "estimator": MultinomialNB(),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__alpha": [0, 0.1, 0.5, 1]
    }
}


# In[ ]:


mnb_count_params = {
    "name": "mnb_count",
    "consolidate_text": False,
    "vec": CountVectorizer(),
    "estimator": MultinomialNB(),
    "params": {
        "vec__stop_words": [None, "english", new_stop_words],
        "vec__min_df": [1, 2, 5],
        "vec__max_features": [None, 5000, 10000],
        "vec__binary": [False, True],
        "estimator__alpha": [0, 0.1, 0.5, 1]
    }
}


# In[ ]:





# ### Random Forest Classifier Models

# In[ ]:





# In[ ]:





# In[ ]:





# ### Ada Boost Classifier Models

# In[ ]:





# ### Support Vector Machine Models

# In[ ]:


svc_pipe = Pipeline([
    ("svc_ss", StandardScaler()),
    ("svc", SVC()),
])

svc_params = {
    
}


# In[ ]:





# In[ ]:





# In[ ]:




