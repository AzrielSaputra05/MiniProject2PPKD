import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from essentials import load_data
import plotly.express as px

st.set_page_config(
    page_title="Visualisasi Dataset",
    page_icon="📈",
    layout="wide"
)

df = load_data()

st.title("📈 Visualisasi dan Analisis Dataset")

# Sidebar untuk filter
st.sidebar.header("🔍 Filter Data")
if 'Kategori' in df.columns:
    kategori_options = ['Semua'] + df['Kategori'].unique().tolist()
    selected_kategori = st.sidebar.selectbox("Pilih Kategori:", kategori_options)
    
    if selected_kategori != 'Semua':
        df_filtered = df[df['Kategori'] == selected_kategori]
    else:
        df_filtered = df
else:
    df_filtered = df

# Tabs untuk berbagai visualisasi
tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi", "📉 Scatter Plot", "🔥 Heatmap", "📈 Time Series"])

with tab1:
    st.subheader("Distribusi Total Revenue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df_filtered["Total"], bins=30, edgecolor='black', alpha=0.7)
        ax.set_title("Distribusi Total Revenue", fontsize=14)
        ax.set_xlabel("Total Revenue", fontsize=12)
        ax.set_ylabel("Frekuensi", fontsize=12)
        st.pyplot(fig)
    
    with col2:
        # Boxplot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.boxplot(df_filtered["Total"])
        ax.set_title("Boxplot Total Revenue", fontsize=14)
        ax.set_ylabel("Total Revenue", fontsize=12)
        st.pyplot(fig)

with tab2:
    st.subheader("Hubungan Antar Variabel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scatter Plot: Harga Produk vs Total**")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(df_filtered["Harga Produk"], df_filtered["Total"], alpha=0.5)
        ax.set_xlabel("Harga Produk", fontsize=12)
        ax.set_ylabel("Total Revenue", fontsize=12)
        st.pyplot(fig)
    
    with col2:
        st.write("**Scatter Plot: Jumlah Produk vs Total**")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(df_filtered["Jumlah Produk"], df_filtered["Total"], alpha=0.5)
        ax.set_xlabel("Jumlah Produk", fontsize=12)
        ax.set_ylabel("Total Revenue", fontsize=12)
        st.pyplot(fig)

with tab3:
    st.subheader("Correlation Heatmap")
    
    # Select only numeric columns
    num_df = df_filtered.select_dtypes(include=['float64', 'int64'])
    
    if len(num_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            num_df.corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax
        )
        ax.set_title("Correlation Matrix", fontsize=14)
        st.pyplot(fig)
        
        # Tampilkan korelasi tertinggi
        corr_matrix = num_df.corr()
        high_corr = corr_matrix.unstack().sort_values(ascending=False)
        high_corr = high_corr[high_corr < 1]  # Hapus korelasi dengan diri sendiri
        
        st.subheader("Korelasi Tertinggi")
        st.write(high_corr.head(10))
    else:
        st.warning("Tidak cukup kolom numerik untuk membuat heatmap")

with tab4:
    st.subheader("Time Series Analysis")
    
    # Check if there's a date column
    date_cols = [col for col in df_filtered.columns if 'tanggal' in col.lower() or 'waktu' in col.lower() or 'date' in col.lower()]
    
    if date_cols:
        # Use plotly for interactive time series
        date_col = date_cols[0]  # Use the first date column
        fig = px.line(df_filtered, x=date_col, y='Total', title='Revenue Over Time')
        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Total Revenue",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada kolom tanggal dalam dataset. Tampilkan bar chart berdasarkan bulan.")
        
        # Alternative: Show by month if available
        if 'bulan' in df_filtered.columns:
            monthly_revenue = df_filtered.groupby('bulan')['Total'].mean().reset_index()
            fig = px.bar(monthly_revenue, x='bulan', y='Total', 
                         title='Rata-rata Revenue per Bulan')
            fig.update_layout(
                xaxis_title="Bulan",
                yaxis_title="Rata-rata Revenue"
            )
            st.plotly_chart(fig, use_container_width=True)

# Info tambahan
with st.expander("ℹ️ Informasi Dataset"):
    st.write("**Jumlah Data:**", len(df_filtered))
    st.write("**Jumlah Kolom:**", len(df_filtered.columns))
    st.write("**Statistik Deskriptif:**")
    st.dataframe(df_filtered.describe())
    
    st.write("**Missing Values:**")
    st.write(df_filtered.isnull().sum())