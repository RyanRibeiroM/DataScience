import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def train_RandomForestClassifier_grid(X_train, Y_train):
    pipeline = Pipeline([
    ('sampler', None),
    ('clf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'sampler':[
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],

        'clf__max_depth': [5,10,20],
        'clf__n_estimators':[100, 200],
        'clf__class_weight': ['balanced', None]
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train,Y_train)

    return grid

def train_LogisticRegression_grid(X_train, Y_train):
    pipeline = Pipeline([
    ('sampler', None),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    param_grid = {
        'sampler': [
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],
        'clf__C': [0.1,1,10],
        'clf__solver': ['lbfgs','newton-cg'],
        'clf__class_weight': ['balanced', None]
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid

def train_SVC_grid(X_train, Y_train):
    pipeline = Pipeline([
        ('sampler', None),
        ('clf', SVC(random_state=42)),
    ])

    param_grid = {
        'sampler':[
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],

        'clf__C': [0.1,1,10],
        'clf__kernel': ['poly', 'rbf', 'sigmoid'],
        'clf__gamma':['scale','auto'],
        'clf__class_weight': ['balanced', None]
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid

def train_GradientBoostingClassifier_grid(X_train, Y_train):
    pipeline = Pipeline([
        ('sampler', None),
        ('clf', GradientBoostingClassifier(random_state=42)),
    ])

    param_grid = {
        'sampler':[
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],

        'clf__learning_rate': [0.01,0.1,0.2],
        'clf__n_estimators': [100,200],
        'clf__max_depth': [3,5,8],
        'clf__subsample': [0.8,1]
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid

def train_DecisionTreeClassifier_grid(X_train, Y_train):
    pipeline = Pipeline([
        ('sampler', None),
        ('clf', DecisionTreeClassifier(random_state=42)),
    ])

    param_grid = {
        'sampler':[
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],

        'clf__max_depth': [5,10,20, 30],
        'clf__criterion': ['gini', 'entropy'],
        'clf__class_weight': ['balanced', None]
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid

def train_MLPClassifier_grid(X_train, Y_train):
    pipeline = Pipeline([
        ('sampler', None),
        ('clf', MLPClassifier(random_state=42, max_iter=1000)),
    ])

    param_grid = {
        'sampler':[
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],

        'clf__activation': ['relu', 'tanh'],
        'clf__hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
        'clf__learning_rate_init': [0.001, 0.01],
        'clf__alpha': [0.0001, 0.001]
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid

def train_KNeighborsClassifier_grid(X_train, Y_train):
    pipeline = Pipeline([
        ('sampler', None),
        ('clf', KNeighborsClassifier()),
    ])

    param_grid = {
        'sampler':[
            SMOTE(random_state=42),
            RandomUnderSampler(random_state=42),
            None
        ],

        'clf__n_neighbors': [3, 5, 7, 11],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, Y_train)
    return grid