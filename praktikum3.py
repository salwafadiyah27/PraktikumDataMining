import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cek apakah file ada ---
file_path = 'covid_19_indonesia_time_series_all.csv'

if not os.path.exists(file_path):
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan skrip ini.")
else:
    # --- Load Dataset ---
    data = pd.read_csv(file_path)

    # --- Preprocessing ---
    # Mengisi missing values jika ada (simple fillna dengan median)
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Jika kolom tidak ditemukan, tampilkan error
    required_columns = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']
    for col in required_columns:
        if col not in data.columns:
            st.error(f"Kolom '{col}' tidak ditemukan dalam dataset.")
            st.stop()

    # Membuat fitur baru 'Fatality Rate'
    data['Fatality Rate'] = np.where(data['Total Cases'] > 0,
                                     data['Total Deaths'] / data['Total Cases'], 0)

    # Fitur numerik yang akan dipakai
    features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Fatality Rate']

    # Scaling fitur numerik
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[features] = scaler.fit_transform(data[features])

    # --- Supervised Learning: Prediksi Total Cases ---
    X = data_scaled[features]
    y = data_scaled['Total Cases']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --- Unsupervised Learning: Clustering dengan KMeans ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    data_scaled['Cluster'] = kmeans.fit_predict(data_scaled[features])

    # --- Streamlit Dashboard ---
    st.title("COVID-19 Indonesia Dashboard")

    st.header("Model Prediksi Total Kasus COVID-19")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    st.header("Clustering Lokasi Berdasarkan Fitur COVID-19")
    cluster_counts = data_scaled['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.header("Visualisasi Cluster dengan Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data_scaled, x='Population Density', y='Total Cases', hue='Cluster', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.header("Data Lokasi dengan Cluster")
    st.dataframe(data_scaled[['Location', 'Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density', 'Fatality Rate', 'Cluster']].head(20))

    # --- Mockup: Ringkasan Risiko ---
    st.header("Ringkasan Tingkat Risiko Wilayah")
    risk_summary = data_scaled.groupby('Cluster').agg({
        'Total Cases': 'mean',
        'Fatality Rate': 'mean'
    }).rename(columns={'Total Cases': 'Avg Total Cases', 'Fatality Rate': 'Avg Fatality Rate'})

    st.table(risk_summary)
