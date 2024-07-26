import warnings
import os
import numpy as np
import pandas as pd
import pickle
import pandas as pd
import copy
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Sample: clinical model
data = pd.read_excel('data/clin.xlsx')
df = pd.DataFrame(data)
workbook_dir = './workbook/dcis'
os.makedirs(workbook_dir, exist_ok=True)

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

# Define classifiers and their parameters
classifiers = {
    'KNN': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance'),
    'Logistic Regression': LogisticRegression(C=0.001, penalty='l2', solver='liblinear'),
    'Random Forest': RandomForestClassifier(n_estimators=8),
    'Decision Tree': tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=2),
    'AdaBoost': AdaBoostClassifier(learning_rate=0.1, n_estimators=100),
    'SVC': SVC(C=10, gamma=0.01, kernel='rbf', probability=True),
    'Extra Trees': ExtraTreesClassifier(max_features='log2', n_estimators=200, criterion='gini', max_depth=8, min_samples_split=2),
    'XGBoost': XGBClassifier(n_estimators=500, max_depth=5, learning_rate=1, colsample_bytree=0.5, subsample=0.5),
    'MLP': MLPClassifier(hidden_layer_sizes=(50, 50), activation='tanh', solver='sgd', alpha=0.1)
}

# Initialize variables for best model
best_auc = 0.5
best_model_name = None

# Evaluation and model selection with Stratified KFold
skf = StratifiedKFold(n_splits=5)  # Adjust n_splits as needed

for model_name, model in classifiers.items():
    all_results = []  # Store results for each fold
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        auc = roc_auc_score(y_test_fold, model.predict_proba(X_test_fold)[:, 1])
        all_results.append(auc)

    avg_auc = np.mean(all_results)
    if avg_auc > best_auc:
        best_auc = avg_auc
        best_model_name = model_name
        best_model = copy.deepcopy(model)

# Save best model
with open('./models/model1.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Prepare results DataFrame
results = []
for model_name, model in classifiers.items():
    if model_name == best_model_name:
        y_pred = model.predict(X_test)
        all_metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": best_auc  # Use the best AUC obtained from cross-validation
        }
    else:
        all_metrics = {"Model": model_name, "NA": "Not Selected"}
    results.append(all_metrics)

df_model = pd.DataFrame(results)
df_model.to_excel(os.path.join(workbook_dir, 'model1.xlsx'), index=False)

# Print best model performance
print(f"Best Model: {best_model_name}")
print(f"Best AUC: {best_auc:.4f}")