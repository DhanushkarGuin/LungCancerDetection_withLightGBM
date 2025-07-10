import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset_med.csv')

# print(dataset.isnull().sum()) # No missing values

dataset.drop(['id'], axis=1, inplace=True)
dataset.drop(['country'], axis=1, inplace=True)

# print(dataset.head())

## Converting Date Strings to Numerics
dataset['diagnosis_date'] = pd.to_datetime(dataset['diagnosis_date'], errors='coerce')
dataset['end_treatment_date'] = pd.to_datetime(dataset['end_treatment_date'], errors='coerce')

dataset['diagnosis_year'] = dataset['diagnosis_date'].dt.year
dataset['diagnosis_month'] = dataset['diagnosis_date'].dt.month
dataset['diagnosis_day'] = dataset['diagnosis_date'].dt.day

dataset['end_treatment_year'] = dataset['end_treatment_date'].dt.year
dataset['end_treatment_month'] = dataset['end_treatment_date'].dt.month
dataset['end_treatment_day'] = dataset['end_treatment_date'].dt.day

dataset.drop(['diagnosis_date', 'end_treatment_date'], axis=1, inplace=True)

X = dataset.drop(columns=['survived'])
y = dataset['survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(dataset.columns.tolist())

categorical_features = ['gender', 'cancer_stage',
                        'family_history', 'smoking_status', 
                        'treatment_type']

numerical_features = ['age', 'bmi', 'cholesterol_level', 
                    'hypertension', 'asthma', 'cirrhosis', 
                    'other_cancer','diagnosis_year', 
                    'diagnosis_month', 
                    'end_treatment_year', 'end_treatment_month']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state = 42, is_unbalance = True))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

from sklearn.metrics import precision_score,recall_score,confusion_matrix
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

## Finding out the important features only
model = pipeline.named_steps['classifier']
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = 10
print("Top features:")
for i in range(top_n):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]}")

# print(dataset.columns.tolist())

## Exporting pipeline with pickle
import pickle

pickle.dump(pipeline,open('pipeline.pkl','wb'))