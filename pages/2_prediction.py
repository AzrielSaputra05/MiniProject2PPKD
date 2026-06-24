import streamlit as st
import pandas as pd
import numpy as np
from essentials import (
    load_data, 
    load_ridge_model, 
    load_ann_model, 
    load_scaler, 
    load_encoder,
    predict_revenue_ridge,
    predict_revenue_ann,
    get_models
)
from datetime import datetime

st.set_page_config(
    page_title="Prediksi Revenue",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Prediksi Revenue")
st.write("Prediksi total revenue menggunakan Machine Learning")

# Load data and models
df = load_data()
model_ridge, model_ann, scaler, encoder = get_models()

# Sidebar untuk pilihan model
st.sidebar.header("⚙️ Pengaturan Model")
model_type = st.sidebar.selectbox(
    "Pilih Model",
    ["Ridge Regression", "Artificial Neural Network (ANN)"]
)

# Input features
st.subheader("📝 Input Data Prediksi")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📅 Waktu**")
    bulan = st.number_input("Bulan (1-12):", min_value=1, max_value=12, value=1)
    tahun = st.number_input("Tahun:", min_value=2020, max_value=2030, value=2024)
    quarter = st.selectbox("Quarter:", options=[1, 2, 3, 4], index=0)
    
    st.markdown("**🏪 Outlet**")
    outlet = st.selectbox(
        "Pilih Outlet:",
        options=["AYAM SERAYU - CABANG 1", "AYAM SERAYU - CABANG 2", "AYAM SERAYU - PUSAT"],
        index=0
    )

with col2:
    st.markdown("**📈 Lag Features**")
    lag_1 = st.number_input(
        "Lag 1 (Revenue bulan sebelumnya):", 
        min_value=0.0, 
        value=300000000.0, 
        step=10000000.0,
        format="%.0f"
    )
    lag_3 = st.number_input(
        "Lag 3 (Revenue 3 bulan lalu):", 
        min_value=0.0, 
        value=290000000.0, 
        step=10000000.0,
        format="%.0f"
    )
    lag_6 = st.number_input(
        "Lag 6 (Revenue 6 bulan lalu):", 
        min_value=0.0, 
        value=280000000.0, 
        step=10000000.0,
        format="%.0f"
    )
    rolling_3 = st.number_input(
        "Rolling 3 (Rata-rata 3 bulan):", 
        min_value=0.0, 
        value=295000000.0, 
        step=10000000.0,
        format="%.0f"
    )

# Tombol prediksi
if st.button("🔮 Prediksi Revenue", use_container_width=True, type="primary"):
    
    if model_type == "Ridge Regression":
        if model_ridge is None:
            st.error("Model Ridge tidak ditemukan. Pastikan file 'prediction_ridge.joblib' ada.")
        else:
            try:
                prediction = predict_revenue_ridge(
                    outlet=outlet,
                    bulan=bulan,
                    tahun=tahun,
                    quarter=quarter,
                    lag_1=lag_1,
                    lag_3=lag_3,
                    lag_6=lag_6,
                    rolling_3=rolling_3
                )
                
                # Tampilkan hasil
                st.divider()
                st.subheader("📊 Hasil Prediksi")
                
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.metric("📊 Model", "Ridge Regression")
                with col_result2:
                    st.metric("💰 Prediksi Revenue", f"Rp {prediction:,.0f}")
                with col_result3:
                    future_pred = prediction * 6
                    st.metric("📈 Revenue 6 Bulan", f"Rp {future_pred:,.0f}")
                
                with st.expander("📋 Detail Prediksi", expanded=False):
                    st.write("**Input Features:**")
                    input_data = pd.DataFrame([{
                        'Bulan': bulan,
                        'Tahun': tahun,
                        'Quarter': quarter,
                        'Outlet': outlet,
                        'Lag 1': f"Rp {lag_1:,.0f}",
                        'Lag 3': f"Rp {lag_3:,.0f}",
                        'Lag 6': f"Rp {lag_6:,.0f}",
                        'Rolling 3': f"Rp {rolling_3:,.0f}"
                    }])
                    st.dataframe(input_data, use_container_width=True)
                    st.write(f"**Hasil Prediksi:** Rp {prediction:,.0f}")
                    
            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
                st.info("Pastikan input features sesuai dengan yang diharapkan model")
    
    elif model_type == "Artificial Neural Network (ANN)":
        if model_ann is None:
            st.error("Model ANN tidak ditemukan. Pastikan file 'ann_model.keras' ada.")
        else:
            try:
                prediction = predict_revenue_ann(
                    outlet=outlet,
                    bulan=bulan,
                    tahun=tahun,
                    quarter=quarter,
                    lag_1=lag_1,
                    lag_3=lag_3,
                    lag_6=lag_6,
                    rolling_3=rolling_3
                )
                
                st.divider()
                st.subheader("📊 Hasil Prediksi")
                
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.metric("📊 Model", "ANN")
                with col_result2:
                    st.metric("💰 Prediksi Revenue", f"Rp {prediction:,.0f}")
                with col_result3:
                    future_pred = prediction * 6
                    st.metric("📈 Revenue 6 Bulan", f"Rp {future_pred:,.0f}")
                
                with st.expander("📋 Detail Prediksi", expanded=False):
                    st.write("**Input Features:**")
                    input_data = pd.DataFrame([{
                        'Bulan': bulan,
                        'Tahun': tahun,
                        'Quarter': quarter,
                        'Outlet': outlet,
                        'Lag 1': f"Rp {lag_1:,.0f}",
                        'Lag 3': f"Rp {lag_3:,.0f}",
                        'Lag 6': f"Rp {lag_6:,.0f}",
                        'Rolling 3': f"Rp {rolling_3:,.0f}"
                    }])
                    st.dataframe(input_data, use_container_width=True)
                    st.write(f"**Hasil Prediksi:** Rp {prediction:,.0f}")
                    
            except Exception as e:
                st.error(f"Error dalam prediksi ANN: {str(e)}")
                st.info("Pastikan input features sesuai dengan yang diharapkan model")

# Informasi tambahan di sidebar
with st.sidebar:
    st.divider()
    st.subheader("📊 Statistik Dataset")
    
    if df is not None:
        st.write(f"Total Data: {len(df):,}")
        st.write(f"Kolom: {len(df.columns)}")
        
        if 'Total' in df.columns:
            st.write(f"Revenue Rata-rata: Rp {df['Total'].mean():,.0f}")
            st.write(f"Revenue Max: Rp {df['Total'].max():,.0f}")
            st.write(f"Revenue Min: Rp {df['Total'].min():,.0f}")
    
    st.divider()
    st.caption("Made with ❤️ by Data Science Team")