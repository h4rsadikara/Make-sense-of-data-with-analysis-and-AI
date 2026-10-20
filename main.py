"""Studi Kasus 1 - Make sense of data with analysis and AI.
Kasus: Pemantauan PM2.5 harian (data sintetis) dengan pola musim kemarau/kabut asap.
Alur: data -> EDA -> deteksi anomali (AI) -> prediksi sederhana -> insight.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
days = pd.date_range("2025-01-01", periods=365)
season = 25 + 30 * np.exp(-((days.dayofyear - 250) ** 2) / (2 * 30 ** 2))  # puncak ~Sept
temp = 27 + 3 * np.sin(2 * np.pi * days.dayofyear / 365) + rng.normal(0, 0.8, 365)
hotspot = rng.poisson(season / 5)
pm25 = season + 1.5 * hotspot + rng.normal(0, 5, 365)
df = pd.DataFrame({"tanggal": days, "pm25": pm25, "suhu": temp, "hotspot": hotspot})

# 1. EDA
print("== Statistik deskriptif ==")
print(df[["pm25", "suhu", "hotspot"]].describe().round(2))
print("\n== Korelasi ==")
print(df[["pm25", "suhu", "hotspot"]].corr().round(2))

# 2. Deteksi anomali dengan Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
df["anomali"] = iso.fit_predict(df[["pm25", "hotspot"]]) == -1
print(f"\nJumlah hari anomali: {df.anomali.sum()}")

# 3. Prediksi PM2.5 dari hotspot + suhu
X, y = df[["hotspot", "suhu"]], df["pm25"]
model = LinearRegression().fit(X, y)
print(f"R^2 model regresi: {model.score(X, y):.3f}")

# 4. Visualisasi
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.tanggal, df.pm25, lw=1, label="PM2.5")
ax.scatter(df.tanggal[df.anomali], df.pm25[df.anomali], c="red", s=18, label="Anomali")
ax.axhline(55.5, ls="--", c="gray", label="Ambang tidak sehat (55.5)")
ax.set(title="PM2.5 harian & anomali (data sintetis)", ylabel="ug/m3"); ax.legend()
fig.tight_layout(); fig.savefig("outputs/case1_pm25.png", dpi=150)
print("Gambar tersimpan: outputs/case1_pm25.png")
