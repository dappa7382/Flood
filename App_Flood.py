# Modern Flood Dashboard Layout
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# Load model & data
reg_model = joblib.load('flood_probability_regressor_new.pkl')
clf_model = joblib.load('flood_risk_classifier.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('flood.csv')

# Selected features
selected_features = [
    'MonsoonIntensity', 'TopographyDrainage', 'Deforestation',
    'Urbanization', 'Encroachments', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'PopulationScore', 'WetlandLoss'
]

# Add clustering
X_scaled = scaler.transform(df[selected_features])
cluster_labels = kmeans_model.predict(X_scaled)
df['FloodRiskCluster'] = cluster_labels

# Sidebar
st.sidebar.title("🌊 Flood Risk Dashboard")
page = st.sidebar.radio("📌 Menu", [
    "Dashboard Utama",
    "Prediksi Risiko Banjir",
    "Visualisasi Data",
    "Clustering Daerah"
])

# Dashboard Utama
if page == "Dashboard Utama":
    st.title("🌟 Dashboard Risiko Banjir")

    st.markdown("### 👋 Selamat Datang di Flood Risk Dashboard")

    # Top KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Data", f"{len(df)} daerah")
    col2.metric("Mean Flood Probability", f"{df['FloodProbability'].mean():.2f}")
    col3.metric("Max Flood Probability", f"{df['FloodProbability'].max():.2f}")
    col4.metric("Jumlah Cluster", f"{df['FloodRiskCluster'].nunique()} cluster")

    st.markdown("---")

    # KPI Chart 1 → Pie chart Cluster distribution
    st.subheader("📊 Distribusi Cluster Risiko Banjir")
    cluster_counts = df['FloodRiskCluster'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # KPI Chart 2 → Trend FloodProbability
    st.subheader("📈 Trend Flood Probability (Sorted Index)")
    df_sorted = df.sort_values(by='FloodProbability').reset_index()
    fig2, ax2 = plt.subplots()
    ax2.plot(df_sorted.index, df_sorted['FloodProbability'], color='blue', linewidth=2)
    ax2.set_xlabel("Daerah (sorted index)")
    ax2.set_ylabel("Flood Probability")
    st.pyplot(fig2)

    # Data Table → Daerah Top 10
    st.subheader("📋 Daerah dengan Flood Probability Tertinggi")
    st.dataframe(df[selected_features + ['FloodProbability', 'FloodRiskCluster']].sort_values(by='FloodProbability', ascending=False).head(10))

# Prediksi Risiko Banjir
elif page == "Prediksi Risiko Banjir":
    st.title("🚀 Prediksi Risiko Banjir")

    st.subheader("Masukkan nilai fitur:")
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

    input_df = pd.DataFrame([input_data])
    X_input_scaled = scaler.transform(input_df)

    flood_prob_pred = reg_model.predict(X_input_scaled)[0]
    flood_risk_class_pred = clf_model.predict(X_input_scaled)[0]

    risk_labels = ['Low', 'Medium', 'High']
    risk_pred_label = risk_labels[flood_risk_class_pred]

    st.subheader("Hasil Prediksi:")
    st.write(f"**Probabilitas Banjir:** {flood_prob_pred:.2f}")
    st.write(f"**Kategori Risiko:** {risk_pred_label}")

# Visualisasi Data
elif page == "Visualisasi Data":
    st.title("📊 Visualisasi Data")

    st.subheader("Correlation Matrix")
    plt.figure(figsize=(12, 10))
    corr_matrix = df[selected_features + ['FloodProbability']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Clustering Daerah
elif page == "Clustering Daerah":
    st.title("🔍 Clustering Daerah Berdasarkan Faktor Risiko Banjir")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.subheader("Visualisasi Cluster (PCA 2D)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['FloodRiskCluster'], palette='Set2', s=100)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    st.pyplot(fig)

    st.subheader("Daftar Daerah dengan Cluster")
    st.dataframe(df[selected_features + ['FloodProbability', 'FloodRiskCluster']].head(20))

    with st.expander("📋 Profil Rata-rata per Cluster"):
        cluster_profile = df.groupby('FloodRiskCluster')[selected_features + ['FloodProbability']].mean()
        st.dataframe(cluster_profile)
