import streamlit as st
import pandas as pd
from essentials import load_data

st.set_page_config(
    page_title="Visualisasi Dataset",
    page_icon="📊",
    layout="wide"
)

df = load_data()

st.title("📊 Visualisasi Dataset")
st.write("Dataset Revenue Ayam Serayu")

# Tabs untuk berbagai tampilan
tab1, tab2, tab3 = st.tabs(["📋 Preview Data", "ℹ️ Informasi", "📈 Statistik"])

with tab1:
    st.subheader("Preview Dataset")
    st.dataframe(df.head(10))
    
    # Pilihan jumlah data yang ditampilkan
    n_rows = st.slider("Jumlah baris yang ditampilkan:", 5, 50, 10)
    st.dataframe(df.head(n_rows))

with tab2:
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Shape:**", df.shape)
        st.write("**Jumlah Baris:**", len(df))
        st.write("**Jumlah Kolom:**", len(df.columns))
    
    with col2:
        st.write("**Tipe Data:**")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")
    
    st.subheader("Daftar Kolom")
    st.write(df.columns.tolist())
    
    st.subheader("Informasi Null Values")
    st.write(df.isnull().sum())

with tab3:
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"Terdapat {missing.sum()} data yang hilang di dataset")