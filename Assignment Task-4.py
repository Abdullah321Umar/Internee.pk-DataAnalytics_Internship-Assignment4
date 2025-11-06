"""
Fraud Detection in Applications (Assignment 4) """

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# ---------------- Config ----------------
DATA_PATH = "C:/Users/Abdullah Umer/Desktop/Internee.pk Internship/Task 4/application_data.xlsx"     
OUTPUT_FLAGGED = "flagged_applications.csv"
IMG_DIR = "vis_images"
ARTIFACT_DIR = "artifacts"
RANDOM_STATE = 42
CONTAMINATION = 0.01       # expected fraction of anomalies (tune if you know expected fraud rate)
SAMPLE_FOR_FIT = None      # None => use full dataset (small file), or set e.g. 20000 for large

plt.style.use('dark_background')  # make all plots dark
PALETTE = ['#00E5FF', '#FFEA00', '#FF4081', '#7C4DFF', '#00E676', '#FF6D00', '#00B0FF', '#FFD740']

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- Load ----------------
print("Loading:", DATA_PATH)
# Read excel (first sheet)
df = pd.read_excel(DATA_PATH, engine='openpyxl')
print("Shape:", df.shape)
print("Columns preview:", df.columns.tolist()[:15])

# ---------------- Choose / adapt columns ----------------
# Typical fields from your original dataset. Edit here if your smaller file uses other names.
candidate_cols = [
    'SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_CONTRACT_TYPE'
]
cols = [c for c in candidate_cols if c in df.columns]
data = df[cols].copy()
print("Using columns:", cols)

# ---------------- Feature engineering ----------------
# Age in years from DAYS_BIRTH (dataset has negative days)
if 'DAYS_BIRTH' in data.columns:
    data['AGE_YEARS'] = (-data['DAYS_BIRTH']) // 365

# Employment years (handle sentinel 365243 => NaN)
if 'DAYS_EMPLOYED' in data.columns:
    data['DAYS_EMPLOYED_CLIP'] = data['DAYS_EMPLOYED'].replace({365243: np.nan})
    data['YEARS_EMPLOYED'] = data['DAYS_EMPLOYED_CLIP'] / 365
else:
    data['YEARS_EMPLOYED'] = np.nan

# Derived numeric features
data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL'].replace(0, np.nan)
data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL'].replace(0, np.nan)
if 'AMT_GOODS_PRICE' in data.columns:
    data['GOODS_CREDIT_DIFF'] = data['AMT_GOODS_PRICE'] - data['AMT_CREDIT']

# Simple rules for suspicious values
data['UNREALISTIC_AGE'] = ((data['AGE_YEARS'] < 16) | (data['AGE_YEARS'] > 75)).astype(int)
data['NEGATIVE_INCOME'] = (data['AMT_INCOME_TOTAL'] <= 0).astype(int)

# ---------------- Modeling features ----------------
model_features = [
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE' if 'AMT_GOODS_PRICE' in data.columns else None,
    'AGE_YEARS', 'YEARS_EMPLOYED', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
    'UNREALISTIC_AGE', 'NEGATIVE_INCOME'
]
# keep only existing
model_features = [f for f in model_features if f is not None and f in data.columns]
print("Model features:", model_features)

# Fill missing numeric with medians
for f in model_features:
    if data[f].isnull().any():
        med = data[f].median()
        data[f] = data[f].fillna(med)

X = data[model_features].astype(float).values

# ---------------- Scale ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Fit IsolationForest + KMeans ----------------
n_rows = X_scaled.shape[0]
print("Rows for modeling:", n_rows)

# Optionally subsample for model fitting to speed up (SAMPLE_FOR_FIT)
if SAMPLE_FOR_FIT and SAMPLE_FOR_FIT < n_rows:
    rng = np.random.default_rng(RANDOM_STATE)
    idx_sample = rng.choice(n_rows, size=SAMPLE_FOR_FIT, replace=False)
    X_fit = X_scaled[idx_sample]
else:
    X_fit = X_scaled

# IsolationForest
iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                      random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(X_fit)
iso_preds = iso.predict(X_scaled)   # -1 anomaly, 1 normal
data['iso_anomaly'] = (iso_preds == -1).astype(int)
data['iso_score'] = iso.decision_function(X_scaled)  # larger => more normal

# KMeans: choose k using silhouette on a small range
best_k = 4
best_score = -1
for k in range(2, 7):
    km_try = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
    labels_try = km_try.fit_predict(X_fit)
    try:
        s = silhouette_score(X_fit, labels_try)
    except Exception:
        s = -1
    if s > best_score:
        best_score = s
        best_k = k

km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
km.fit(X_fit)
# assign labels for all rows by predicting the nearest center
km_labels = km.predict(X_scaled)
data['kmeans_label'] = km_labels
cluster_counts = data['kmeans_label'].value_counts(normalize=True)
small_clusters = cluster_counts[cluster_counts < 0.03].index.tolist()
data['kmeans_anomaly'] = data['kmeans_label'].apply(lambda x: 1 if x in small_clusters else 0)

# ---------------- Combine anomaly signals ----------------
data['anomaly_score'] = data['iso_anomaly'] + data['kmeans_anomaly'] + data['UNREALISTIC_AGE'] + data['NEGATIVE_INCOME']
data['Alert'] = data['anomaly_score'].apply(lambda v: 'Suspicious' if v >= 1 else 'Normal')

# Save flagged rows
flagged = data[data['Alert'] == 'Suspicious'].copy()
if 'SK_ID_CURR' in df.columns:
    flagged_out = flagged.merge(df, on='SK_ID_CURR', how='left')
else:
    flagged_out = flagged
flagged_out.to_csv(OUTPUT_FLAGGED, index=False)
print("Flagged rows saved to:", OUTPUT_FLAGGED, "count:", len(flagged_out))

# ---------------- PCA for visuals ----------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_2 = pca.fit_transform(X_scaled)
data['pca1'] = pca_2[:, 0]
data['pca2'] = pca_2[:, 1]

# ---------------- Visualization helper ----------------
def save_fig(fig, fname):
    path = os.path.join(IMG_DIR, fname)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", path)



# Visualizations

# 1) Age distribution
fig = plt.figure(figsize=(8,5))
plt.hist(data['AGE_YEARS'].dropna(), bins=40, color=PALETTE[0], alpha=0.95)
plt.title("Age Distribution")
plt.xlabel("Age (years)")
plt.ylabel("Count")
save_fig(fig, "viz1_age_hist.png")

# 2) Income distribution (log)
fig = plt.figure(figsize=(8,5))
vals = data['AMT_INCOME_TOTAL'].replace(0, np.nan).dropna()
plt.hist(np.log1p(vals), bins=50, color=PALETTE[1], alpha=0.95)
plt.title("Log Income Distribution")
plt.xlabel("log(1 + income)")
plt.ylabel("Count")
save_fig(fig, "viz2_income_log_hist.png")

# 3) Income vs Credit scatter, anomalies highlighted
fig = plt.figure(figsize=(8,6))
normal = data[data['Alert'] == 'Normal']
susp = data[data['Alert'] == 'Suspicious']
plt.scatter(normal['AMT_INCOME_TOTAL'], normal['AMT_CREDIT'], s=8, alpha=0.6, color=PALETTE[2], label='Normal')
plt.scatter(susp['AMT_INCOME_TOTAL'], susp['AMT_CREDIT'], s=18, alpha=0.9, color=PALETTE[3], label='Suspicious')
plt.xscale('log'); plt.yscale('log')
plt.xlabel("AMT_INCOME_TOTAL (log)")
plt.ylabel("AMT_CREDIT (log)")
plt.title("Income vs Credit (Suspicious highlighted)")
plt.legend()
save_fig(fig, "viz3_income_credit_scatter.png")

# 4) PCA 2D projection colored by Alert
fig = plt.figure(figsize=(8,6))
plt.scatter(normal['pca1'], normal['pca2'], s=10, alpha=0.7, color=PALETTE[0], label='Normal')
plt.scatter(susp['pca1'], susp['pca2'], s=14, alpha=0.9, color=PALETTE[3], label='Suspicious')
plt.xlabel("PCA1"); plt.ylabel("PCA2"); plt.title("PCA Projection")
plt.legend()
save_fig(fig, "viz4_pca.png")

# 5) Boxplot of CREDIT_INCOME_RATIO
fig = plt.figure(figsize=(7,5))
rat = data['CREDIT_INCOME_RATIO'].dropna()
plt.boxplot(rat, vert=True, patch_artist=True,
            boxprops=dict(facecolor=PALETTE[4], color='white'),
            medianprops=dict(color='white'))
plt.title("Credit/Income Ratio (boxplot)")
save_fig(fig, "viz5_credit_income_box.png")

# 6) UNREALISTIC_AGE & NEGATIVE_INCOME counts (side-by-side bars)
fig = plt.figure(figsize=(7,5))
counts_unreal = data['UNREALISTIC_AGE'].value_counts().sort_index()
counts_neg = data['NEGATIVE_INCOME'].value_counts().sort_index()
labels = ['False', 'True']
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, counts_unreal.values, width=width, color=PALETTE[5], label='UNREALISTIC_AGE')
plt.bar(x + width/2, counts_neg.values, width=width, color=PALETTE[6], label='NEGATIVE_INCOME')
plt.xticks(x, labels)
plt.ylabel("Count")
plt.title("Counts of Rule-based Flags")
plt.legend()
save_fig(fig, "viz6_flag_counts.png")

# 7) IsolationForest decision scores distribution
fig = plt.figure(figsize=(7,5))
plt.hist(data['iso_score'], bins=60, color=PALETTE[2], alpha=0.95)
plt.title("IsolationForest Scores (higher = more normal)")
save_fig(fig, "viz7_iso_scores.png")

# 8) KMeans cluster sizes
fig = plt.figure(figsize=(7,5))
cluster_sizes = data['kmeans_label'].value_counts().sort_index()
plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color=PALETTE[:len(cluster_sizes)])
plt.title(f"KMeans Cluster Sizes (k={best_k})")
plt.xlabel("Cluster label")
plt.ylabel("Count")
save_fig(fig, "viz8_kmeans_sizes.png")

# 9) Goods price - credit difference (if exists)
if 'GOODS_CREDIT_DIFF' in data.columns:
    fig = plt.figure(figsize=(7,5))
    plt.hist(data['GOODS_CREDIT_DIFF'].dropna(), bins=60, color=PALETTE[7], alpha=0.95)
    plt.title("AMT_GOODS_PRICE - AMT_CREDIT")
    save_fig(fig, "viz9_goods_credit_diff.png")

# 10) TARGET vs Alert (if TARGET exists)
if 'TARGET' in data.columns:
    fig = plt.figure(figsize=(7,5))
    cross = pd.crosstab(df['TARGET'], data['Alert'])
    ax = cross.plot(kind='bar', stacked=False, color=[PALETTE[0], PALETTE[3]], legend=True)
    ax.set_title("TARGET vs Alert")
    ax.set_xlabel("TARGET")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    save_fig(fig, "viz10_target_vs_alert.png")

# ---------------- Save artifacts ----------------
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
joblib.dump(iso, os.path.join(ARTIFACT_DIR, "isolation_forest.joblib"))
joblib.dump(km, os.path.join(ARTIFACT_DIR, "kmeans.joblib"))
print("Artifacts saved in:", ARTIFACT_DIR)

# ---------------- Summary ----------------
print("Total rows:", len(data))
print("Total flagged (Suspicious):", int((data['Alert'] == 'Suspicious').sum()))
print("Visualizations in:", IMG_DIR)
print("Flagged CSV:", OUTPUT_FLAGGED)











