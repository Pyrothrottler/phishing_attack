# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:32:22 2025

@author: karun
"""

import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import shutil

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"phishing_final.csv")

# Define new feature extraction logic (assuming 'url' column does NOT exist, we build a fake one)
df['constructed_url'] = 'http://example.com/' + df['domaininsubdomains'].astype(str)  # TEMP placeholder

phishing_keywords = ['login', 'signin', 'verify', 'account', 'secure', 'update', 'webscr']
tlds_phishing = ['.ga', '.cf', '.ml', '.tk', '.gq']

def extract_features_from_row(row):
    url = row['constructed_url']
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    return pd.Series({
        'url_length': len(url),
        'numdots': url.count('.'),
        'numdash': url.count('-'),
        'urllength': len(url),
        'domaininsubdomains': row['domaininsubdomains'],
        'domaininpaths': row['domaininpaths'],
        'httpsinhostname': row['httpsinhostname'],
        'hostnamelength': len(hostname),
        'querylength': len(query),
        'pathlength': len(path),
        'randomstring': row['randomstring'],
        'insecureforms': row['insecureforms'],
        'submitinfotoemail': row['submitinfotoemail'],
        'has_login_keyword': 1 if any(word in url.lower() for word in phishing_keywords) else 0,
        'suspicious_tld': 1 if any(url.endswith(tld) for tld in tlds_phishing) else 0,
        'excessive_subdomains': 1 if len(hostname.split('.')) > 4 else 0,
        'label': row['label']
    })

print(" Extracting enhanced features...")
df_features = df.apply(extract_features_from_row, axis=1)

features = [
    'url_length', 'numdots', 'numdash', 'urllength',
    'domaininsubdomains', 'domaininpaths', 'httpsinhostname',
    'hostnamelength', 'querylength', 'pathlength',
    'randomstring', 'insecureforms', 'submitinfotoemail',
    'has_login_keyword', 'suspicious_tld', 'excessive_subdomains'
]

X = df_features[features]
y = df_features['label']

print(" Splitting and training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ✅ Define path to Downloads folder
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
model_filename = "phishing_detector2.pkl"
model_path = os.path.join(downloads_folder, model_filename)

# ✅ Save the trained model directly to Downloads folder
joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")

print(" Saving model as phishing_detector2.pkl")
joblib.dump(model, "phishing_detector2.pkl")
print(" Model saved!")
