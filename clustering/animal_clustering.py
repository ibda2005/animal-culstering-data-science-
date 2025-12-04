# ================================================================
# ANIMAL BEHAVIOR - AGGLOMERATIVE CLUSTERING (VS CODE VERSION)
# ================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ================================================================
# 1. LOAD DATASET
# ================================================================
# Ganti nama file CSV kamu di sini:
filename = "biological_behavior_data.csv"

df = pd.read_csv(filename)
print("=== SAMPLE DATA ===")
print(df.head(), "\n")

# ================================================================
# 2. DATA CLEANING
# ================================================================
print("=== CEK MISSING VALUES ===")
print(df.isnull().sum(), "\n")

df = df.dropna()  # jika mau imputasi, ganti sendiri

print("Setelah cleaning:")
print(df.isnull().sum(), "\n")

# Label encoding kolom kategorikal
cat_cols = ["Species", "ActivityType", "HabitatType"]
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Fitur numerik untuk scaling
num_cols = ["AvgDailyMovement_km", "HeartRate_bpm", "BodyTemp_C", "AggressionLevel"]

scaler = StandardScaler()  # bisa ganti jadi MinMaxScaler()
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

print("=== DATASET SETELAH ENCODING & SCALING ===")
print(df_scaled.head(), "\n")

# ================================================================
# 3. ANALISIS DESKRIPTIF + VISUALISASI
# ================================================================
print("=== STATISTIK DESKRIPTIF ===")
print(df_scaled.describe(), "\n")

# Histogram tiap fitur
df_scaled[num_cols].hist(bins=8, figsize=(12, 6))
plt.suptitle("Histogram Fitur Numerik")
plt.show()

# Heatmap korelasi
plt.figure(figsize=(7, 6))
sns.heatmap(df_scaled[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi")
plt.show()

# ================================================================
# 4. AGGLOMERATIVE CLUSTERING (Single, Complete, Average, Ward)
# ================================================================
X = df_scaled[num_cols]

methods = ["single", "complete", "average", "ward"]
cluster_results = {}

print("=== AGGLOMERATIVE CLUSTERING ===\n")
for m in methods:
    print(f"--- {m.upper()} LINKAGE ---")
    model = AgglomerativeClustering(
        n_clusters=3,  
        linkage=m,
        metric="euclidean"
    )

    labels = model.fit_predict(X)
    df_scaled[f"Cluster_{m}"] = labels
    cluster_results[m] = labels

    print(pd.Series(labels).value_counts())
    print()

# ================================================================
# 5. DENDROGRAM (Ward Linkage)
# ================================================================
plt.figure(figsize=(10, 6))
Z = linkage(X, method="ward", metric="euclidean")
dendrogram(Z)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

print("""
=== PETUNJUK INTERPRETASI DENDROGRAM ===
- Cari garis horizontal paling panjang (largest vertical gap)
- Tarik garis horizontal sebagai threshold
- Hitung berapa cabang (clusters) yang dipotong

Kirim screenshot dendrogram ke saya, nanti saya bantu menentukan cluster ideal.
""")
