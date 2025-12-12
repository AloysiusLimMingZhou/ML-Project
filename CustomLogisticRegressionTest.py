# This is the same thing as SKLearn_LogisticRegressionTest.py, but I use my own custom Logistic Regression model ez

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from CustomLogisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Beginning Step 1: Data Fetching from dataset...")
df = pd.read_csv(r"cleveland.csv", na_values="?") # Treat all values with ? as NaN
df = df.apply(pd.to_numeric, errors="coerce") # Basic categorical conversion, if string then become NaN, not recommended since its better to numerical it
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0) # Since Cleveland is multiclass and I use my regression algorithm to be binary only, so let all be binary (1-4 = 1)
print(np.shape(df))
print("Step 1: Data Fetching from dataset done!")

print("Beginning Step 2: Preprocessing and cleaning data...")
df.dropna(inplace=True)
y = df['num'].values.reshape(-1, 1)
x = df.drop(columns=['num'], axis=1)
print(np.shape(y))
print(np.shape(x))
print("Step 2: Preprocessing and cleaning data done!")

print("Beginning Step 3: Normalizing and Splitting of data...")
x_scaled = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print("Step 3: Normalizing and Splitting of data done!")

print("Beginning Step 4: Setting up Logistic Regression Model and fit...")
model = (LogisticRegression(lr=0.01, l2=0.0001, epoch=2000))
model.fit(x_train, y_train, x_test, y_test)
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)
print("Step 4: Setting up Logistic Regression Model and fit done!")

print("Beginning Step 5: Printing Results Score...")
y_pred_flat = y_pred.ravel()
y_test_flat = y_test.ravel()
y_pred_proba_flat = y_pred_proba.ravel()

accuracy = accuracy_score(y_test_flat, y_pred_flat)
precision = precision_score(y_test_flat, y_pred_flat, average='binary', zero_division=0)
recall = recall_score(y_test_flat, y_pred_flat, average='binary', zero_division=0)
f1 = f1_score(y_test_flat, y_pred_flat, average='binary', zero_division=0)
roc_auc = roc_auc_score(y_test_flat, y_pred_proba_flat)

print("----------------------Metrics----------------------")
print(f"Accuracy: {accuracy*100:.4f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Step 5: Printing Results Score done!")