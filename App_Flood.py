# flood_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# Load model
reg_model = joblib.load('flood_probability_regressor_new.pkl')
clf_model = joblib.load('flood_risk_classifier.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature list
selected_features = [
    'MonsoonIntensity', 'TopographyDrainage', 'Deforestation',
    'Urbanization', 'Encroachments', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'PopulationScore', 'WetlandLoss'
]

# Load data
df = pd.read_csv('flood.csv')

# Sidebar navigation (pakai selectbox)
st.sidebar.title("🌊 Flood Risk Dashboard")
page = st.sidebar.selectbox("📌 Pilih Halaman", [
    "Deskripsi", 
    "Visualisasi Data", 
    "Prediksi Risiko Banjir", 
    "Evaluasi Model", 
    "Clustering Daerah"
])

# Page 1: Deskripsi
if page == "Deskripsi":
    st.title("📖 Deskripsi Dataset & Tujuan Aplikasi")

    with st.expander("🗂️ Lihat Contoh Data (First 5 Rows - Selected Features)"):
        st.dataframe(df[selected_features + ['FloodProbability']].head())

    st.markdown("""
    **Dataset:**  
    Dataset ini berisi faktor-faktor yang memengaruhi risiko banjir di berbagai daerah.

    **Fitur yang digunakan:**  
    - MonsoonIntensity
    - TopographyDrainage
    - Deforestation
    - Urbanization
    - Encroachments
    - DrainageSystems
    - CoastalVulnerability
    - Landslides
    - Watersheds
    - PopulationScore
    - WetlandLoss

    **Target:**  
    - FloodProbability

    **Tujuan Aplikasi:**  
    - Memvisualisasikan data faktor risiko banjir  
    - Memprediksi probabilitas banjir dan klasifikasi risiko banjir  
    - Menyediakan insight bagi perencanaan penanggulangan banjir  
    """)

# Page 2: Visualisasi Data
elif page == "Visualisasi Data":
    st.title("📊 Visualisasi Data")

    tab1, tab2 = st.tabs(["📈 Correlation Matrix", "📉 Distribusi Flood Probability"])

    with tab1:
        st.subheader("Correlation Matrix")
        plt.figure(figsize=(12, 10))
        corr_matrix = df[selected_features + ['FloodProbability']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)

    with tab2:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['FloodProbability'], bins=30, kde=True, color='skyblue', edgecolor='black')
        plt.axvline(df['FloodProbability'].mean(), color='red', linestyle='dashed', linewidth=2)
        plt.text(df['FloodProbability'].mean()+0.01, plt.ylim()[1]*0.9, 'Mean', color='red')
        plt.title("Flood Probability Distribution with Mean Line")
        st.pyplot(plt)

# Page 3: Prediksi Risiko Banjir
elif page == "Prediksi Risiko Banjir":
    st.title("🚀 Prediksi Risiko Banjir")

    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Masukkan nilai fitur:")
        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        input_df = pd.DataFrame([input_data])
        X_input_scaled = scaler.transform(input_df)

    with col_output:
        st.subheader("🔍 Hasil Prediksi:")
        flood_prob_pred = reg_model.predict(X_input_scaled)[0]
        flood_risk_class_pred = clf_model.predict(X_input_scaled)[0]

        risk_labels = ['Low', 'Medium', 'High']
        risk_emojis = ['🟢', '🟡', '🔴']
        risk_pred_label = risk_labels[flood_risk_class_pred]
        risk_emoji = risk_emojis[flood_risk_class_pred]

        st.metric("Probabilitas Banjir", f"{flood_prob_pred:.2f}")
        st.metric("Kategori Risiko", f"{risk_pred_label} {risk_emoji}")

# Page 4: Evaluasi Model
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regression Model (FloodProbability Prediction)")
        st.metric("Mean Squared Error (MSE)", "0.0014")
        st.metric("R² Score", "0.4318")

    with col2:
        st.subheader("Classification Model (FloodRisk Category)")
        st.metric("Accuracy", "0.9822")

    st.markdown("---")

    st.subheader("Feature Importance (FloodProbability Regression)")

    importances = reg_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(8, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title("Feature Importance - FloodProbability Regression")
    st.pyplot(plt)

# Page 5: Clustering Daerah
elif page == "Clustering Daerah":
    st.title("🔍 Clustering Daerah Berdasarkan Faktor Risiko Banjir")

    # Scaling data (INI YANG HARUS ADA supaya X_scaled tidak error!)
    X_scaled = scaler.transform(df[selected_features])

    # Predict cluster untuk seluruh dataset
    cluster_labels = kmeans_model.predict(X_scaled)

    # Tambahkan ke dataframe
    df['FloodRiskCluster'] = cluster_labels

    # Visualisasi PCA 2D
    st.subheader("Visualisasi Cluster (PCA 2D)")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['FloodRiskCluster'], palette='Set2', s=100)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title("Clustering Daerah (PCA 2D)")
    st.pyplot(fig)

    st.markdown("---")

    # Pie chart cluster distribusi
    st.subheader("Distribusi Cluster (Pie Chart 3D-like Effect)")
    cluster_counts = df['FloodRiskCluster'].value_counts().sort_index()

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
                                       startangle=140, explode=explode, shadow=True,
                                       colors=sns.color_palette('Set2'))
    plt.setp(autotexts, size=12, weight='bold')
    ax2.axis('equal')
    st.pyplot(fig2)

    st.markdown("---")

    # Dataframe cluster
    st.subheader("Daftar Daerah dengan Cluster")
    st.dataframe(df[selected_features + ['FloodProbability', 'FloodRiskCluster']].head(20))

    with st.expander("📋 Profil Rata-rata per Cluster"):
        cluster_profile = df.groupby('FloodRiskCluster')[selected_features + ['FloodProbability']].mean()
        st.dataframe(cluster_profile)
