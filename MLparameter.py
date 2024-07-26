import warnings
import random
import pandas as pd
import copy
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# Sample: clinical model
data = pd.read_excel('data/DCISIBCclin.xlsx')
df = pd.DataFrame(data)

# Define exclude and continuous columns
exclude_cols = ['label']  # 'ER', 'PR', 'Her2', 'Ki67'
continuous_cols = [col for col in df.columns if col not in exclude_cols]

# Standardize continuous variables
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Define stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define models and their respective parameter grids for GridSearchCV
models = {
    'SVC': (SVC(), {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]}),
    'AB': (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'],
                                     'metric': ['euclidean', 'manhattan']}),
    'LR': (LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10],
                                  'solver': ['liblinear', 'saga']}),
    'RF': (RandomForestClassifier(), {'n_estimators': [100, 200, 500], 'max_depth': [5, 8, 15],
                                     'min_samples_split': [2, 5, 10]}),
    'DT': (tree.DecisionTreeClassifier(), {'max_depth': [3, 5, 8], 'min_samples_split': [2, 5, 10],
                                     'criterion': ['gini', 'entropy']}),
    'ET': (ExtraTreesClassifier(), {'n_estimators': [100, 200, 500], 'max_depth': [5, 8, 15],
                                     'min_samples_split': [2, 5, 10], 'max_features': ['auto', 'sqrt', 'log2']}),
    'XGB': (XGBClassifier(), {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 7],
                              'learning_rate': [0.1, 0.5, 1], 'subsample': [0.5, 0.8, 1.0],
                              'colsample_bytree': [0.5, 0.8, 1.0]}),
    'MLP': (MLPClassifier(), {'hidden_layer_sizes': [(50,), (100,), (50, 50,)], 'activation': ['tanh', 'relu'],
                              'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05, 0.1]})
}

result_df = pd.DataFrame(columns=['Model', 'Dataset', 'Best Params', 'Best Score'])

# Loop through each model, perform Grid Search, and store the results
for name, (clf, param_grid) in models.items():
    print(f"Tuning {name}...")
    print(f" {name} results:")

    grid = GridSearchCV(clf, param_grid, cv=5)
    grid.fit(X_train, y_train)

    result_df = pd.concat([result_df, pd.DataFrame({
        'Model': name,
        'Dataset': 'Dataset 1',  # You can modify the dataset name if needed
        'Best Params': [grid.best_params_],  # Store best parameters as a list
        'Best Score': [grid.best_score_]  # Store best score as a list
    })], ignore_index=True)

    print("Best Params:", grid.best_params_)
    print("Best Score:", grid.best_score_)
    print("===========================")

result_df.to_excel('output/parameter/tuning_result.xlsx', index=False)