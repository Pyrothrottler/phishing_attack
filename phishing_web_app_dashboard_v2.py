# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:53:56 2025

@author: karun
"""

import streamlit as st
import pandas as pd
import joblib
import re
from urllib.parse import urlparse
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("phishing_detector2.pkl")

# Page setup
st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🔐 Phishing Attack Detection")
st.markdown("Enter a URL manually or upload a CSV for **batch prediction** of Phishing vs Legitimate.")

# Session state tracker
if 'results' not in st.session_state:
    st.session_state.results = []

# Feature extraction logic
def extract_features_from_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    phishing_keywords = ['login', 'signin', 'verify', 'account', 'secure', 'update', 'webscr']
    tlds_phishing = ['.ga', '.cf', '.ml', '.tk', '.gq']

    features = {
        'url_length': len(url),
        'numdots': url.count('.'),
        'numdash': url.count('-'),
        'urllength': len(url),
        'domaininsubdomains': 1 if hostname.count('.') > 2 else 0,
        'domaininpaths': 1 if hostname in path else 0,
        'httpsinhostname': 1 if 'https' in hostname else 0,
        'hostnamelength': len(hostname),
        'querylength': len(query),
        'pathlength': len(path),
        'randomstring': 1 if re.search(r'[a-z]{10,}', hostname) else 0,
        'insecureforms': 0,
        'submitinfotoemail': 1 if 'mailto:' in url else 0,
        'has_login_keyword': 1 if any(k in url.lower() for k in phishing_keywords) else 0,
        'suspicious_tld': 1 if any(url.endswith(tld) for tld in tlds_phishing) else 0,
        'excessive_subdomains': 1 if len(hostname.split('.')) > 4 else 0
    }

    return pd.DataFrame([features])

# --- SINGLE URL CHECK ---
st.markdown("### 🔍 Test a Single URL")
url = st.text_input("Enter the URL to check:")

if url:
    try:
        input_df = extract_features_from_url(url)
        prediction = model.predict(input_df)[0]

        st.session_state.results.append(prediction)

        if prediction == 1:
            st.error("⚠️ This is likely a **Phishing** URL.")
        else:
            st.success("✅ This is likely a **Legitimate** URL.")
    except Exception as e:
        st.warning(f"Error: {e}")

# --- LIVE DASHBOARD ---
st.markdown("---")
st.subheader("📊 Live Detection Dashboard")

phishing_count = st.session_state.results.count(1)
legit_count = st.session_state.results.count(0)

if len(st.session_state.results) > 0:
    fig, ax = plt.subplots()
    ax.pie([phishing_count, legit_count],
           labels=['Phishing', 'Legitimate'],
           colors=['#ff4b4b', '#4bb543'],
           autopct='%1.1f%%', startangle=140)
    ax.axis("equal")
    st.pyplot(fig)
else:
    st.info("No URLs tested yet — enter a URL above or upload a CSV to populate the dashboard.")

st.metric("Total URLs Tested", len(st.session_state.results))
st.metric("Phishing Detected", phishing_count)
st.metric("Legitimate URLs", legit_count)

# --- CSV UPLOAD ---
st.markdown("---")
st.subheader("📁 Batch Test via CSV Upload")

uploaded_file = st.file_uploader("Upload a CSV file with a column named `url`", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        if 'url' not in df_uploaded.columns:
            st.error("⚠️ Your CSV must contain a column named `url`.")
        else:
            with st.spinner("🔎 Analyzing URLs..."):
                features_list = [extract_features_from_url(u).iloc[0] for u in df_uploaded['url']]
                df_features = pd.DataFrame(features_list)
                predictions = model.predict(df_features)
                df_uploaded['prediction'] = predictions
                df_uploaded['prediction'] = df_uploaded['prediction'].map({0: 'Legitimate', 1: 'Phishing'})

            st.success("✅ Analysis complete!")
            st.dataframe(df_uploaded)

            csv = df_uploaded.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Result CSV",
                data=csv,
                file_name="phishing_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.warning(f"❗ Error processing file: {e}")
