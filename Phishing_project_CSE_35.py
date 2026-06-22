#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# In[2]:


# Load UCI dataset (ARFF format)
uci_file_path = r"C:\Users\AKASH S\OneDrive\Documents\Main\Code\Code\Training Dataset.arff"
data, meta = arff.loadarff(uci_file_path)

# Convert ARFF to DataFrame
df_uci = pd.DataFrame(data)

# Decode byte strings if necessary
df_uci = df_uci.map(lambda x: x.decode() if isinstance(x, bytes) else x)

print("UCI dataset loaded successfully!")


# In[3]:


# Load Kaggle dataset
df_kaggle = pd.read_csv(r"C:\Users\AKASH S\OneDrive\Documents\Main\Code\Code\phishing_final.csv")

print("Kaggle dataset loaded successfully!")


# In[4]:


# Convert column names to lowercase & replace spaces with underscores
df_uci.columns = df_uci.columns.str.lower().str.replace(" ", "_")
df_kaggle.columns = df_kaggle.columns.str.lower().str.replace(" ", "_")

# Rename target label column
df_uci.rename(columns={'result': 'label'}, inplace=True)
df_kaggle.rename(columns={'class_label': 'label'}, inplace=True)

# Convert labels to binary (0 = Legitimate, 1 = Phishing)
df_uci['label'] = df_uci['label'].map({-1: 1, 1: 0})  # UCI uses -1 (Phishing), 1 (Legitimate)
df_kaggle['label'] = df_kaggle['label'].astype(int)  # Kaggle dataset already has binary labels

print("Columns standardized and labels converted successfully!")


# In[5]:


# Drop unnecessary columns (like 'id' in Kaggle dataset)
df_kaggle.drop(columns=['id'], errors='ignore', inplace=True)

# Merge datasets, keeping all feature columns
df_combined = pd.concat([df_uci, df_kaggle], ignore_index=True, sort=False)

# Fill missing values
df_combined.fillna(0, inplace=True)  # Replace NaNs with 0

print("Final dataset shape after merging:", df_combined.shape)


# In[13]:


df_combined.to_csv("phishing_final.csv", index=False)
print("Merged dataset saved successfully as phishing_final.csv!")


# In[14]:


# from google.colab import files
# files.download("phishing_final.csv")


# In[15]:


import os
print("Saved files in directory:", os.listdir())


# In[6]:


# Separate features and labels
X = df_combined.drop(columns=['label'], errors='ignore')  # Keep only feature columns
y = df_combined['label']  # Target variable

print("X shape:", X.shape)  # Should contain features now
print("y shape:", y.shape)  # Should contain labels


# In[8]:


# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


# In[9]:


# Apply feature scaling (recommended for better model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling applied!")


# In[10]:


# Initialize and train logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Logistic Regression Performance:\n Accuracy: {accuracy:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1-score: {f1:.4f}")


# In[11]:


# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Random Forest Performance:\n Accuracy: {accuracy_rf:.4f}\n Precision: {precision_rf:.4f}\n Recall: {recall_rf:.4f}\n F1-score: {f1_rf:.4f}")


# In[12]:


# Save the best model (Random Forest) to a file
joblib.dump(rf_model, "phishing_detector.pkl")

print("Model saved successfully as phishing_detector.pkl!")

