import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_FOLDER = "models/preprocess"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Column names
cols = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation',
        'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']

# Load data
df = pd.read_csv("adult.data", header=None, names=cols, na_values='?', skipinitialspace=True)

# Clean: trim + fill missing
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()
    df[c] = df[c].fillna(df[c].mode()[0])
for c in df.select_dtypes(include=['int64','float64']).columns:
    df[c] = df[c].fillna(df[c].median())

# Encode target
df['income'] = df['income'].replace({'>50K.':'>50K','<=50K.':'<=50K'}).str.strip()
y = LabelEncoder().fit_transform(df['income'])
X = df.drop('income', axis=1)

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 5 Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE)
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'classification_report': classification_report(y_test, preds)
    }
    # Save model
    joblib.dump(pipe, os.path.join(SAVE_FOLDER, f"{name}.pkl"))

# Save results to CSV
rows = []
for name in results:
    r = results[name]
    rows.append({
        'model': name,
        'accuracy': r['accuracy'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1']
    })
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,'results_preprocess.csv'), index=False)

print("=== DONE: Model training (With Preprocessing) ===")
for name in results:
    print(f"{name}: Accuracy={results[name]['accuracy']:.4f}, F1={results[name]['f1']:.4f}")
print(f"Models saved in {SAVE_FOLDER}")
print(f"Results CSV saved in {OUTPUT_DIR}/results_preprocess.csv")
