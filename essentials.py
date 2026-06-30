import pandas as pd
import streamlit as st
from joblib import load
import numpy as np
import os

@st.cache_data
def load_data():
    """Load dataset"""
    try:
        possible_paths = [
            "DataSets-PredictRevenue/AyamSerayu_Predict.zip",
            "DataSetsa-PredictRevenue/AyamSerayu_Predict.zip",
            "DataSets-PredictRevenue/AyamSerayu_Predict.csv",
            "DataSetsa-PredictRevenue/AyamSerayu_Predict.csv",
            "AyamSerayu_Predict.zip",
            "AyamSerayu_Predict.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if path.endswith('.zip'):
                    df = pd.read_csv(path, compression="zip")
                else:
                    df = pd.read_csv(path)
                
                if 'Tanggal & Waktu' in df.columns:
                    df['Tanggal & Waktu'] = pd.to_datetime(df['Tanggal & Waktu'], errors='coerce')
                    df = df.dropna(subset=['Tanggal & Waktu'])
                
                print(f"Dataset loaded from: {path}")
                return df
        
        raise FileNotFoundError("Dataset not found")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_ridge_model():
    """Load Ridge Regression model"""
    try:
        possible_paths = [
            "model/prediction_ridge.joblib",
            "model/revenue_prediction.pkl",
            "model/predict.joblib",
            "prediction_ridge.joblib",
            "revenue_prediction.pkl",
            "predict.joblib",
            "app_pipeline.joblib"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data = load(path)
                print(f"✅ Ridge model loaded from: {path}")
                print(f"   Type: {type(data)}")
                
                # Jika data adalah dictionary, cari model di dalamnya
                if isinstance(data, dict):
                    print(f"   Dictionary keys: {list(data.keys())}")
                    for key, value in data.items():
                        if hasattr(value, 'predict'):
                            print(f"   ✅ Found model in dict['{key}']")
                            return value
                    return data
                
                # Jika data adalah pipeline
                elif hasattr(data, 'named_steps'):
                    print(f"   Pipeline steps: {list(data.named_steps.keys())}")
                    for step_name, step_obj in data.named_steps.items():
                        if hasattr(step_obj, 'predict'):
                            print(f"   ✅ Found model in pipeline step: {step_name}")
                            return step_obj
                    return data
                
                # Jika data adalah model langsung
                elif hasattr(data, 'predict'):
                    print(f"   ✅ Object has 'predict' method")
                    return data
                
                else:
                    print(f"   ⚠️ Unknown object type")
                    return data
        
        raise FileNotFoundError("No Ridge model file found")
        
    except Exception as e:
        print(f"❌ Error loading Ridge model: {str(e)}")
        return None

@st.cache_resource
def load_ann_model():
    """Load ANN model"""
    try:
        possible_paths = [
            "model/ann_model.keras",
            "model/ann_model.h5",
            "model/ann_model.joblib",
            "model/ann_preprocessing_pipeline.joblib",
            "ann_model.keras",
            "ann_model.h5",
            "ann_model.joblib",
            "ann_preprocessing_pipeline.joblib"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Coba load sebagai keras model
                try:
                    from tensorflow.keras.models import load_model as load_keras
                    model = load_keras(path)
                    print(f"✅ ANN model (Keras) loaded from: {path}")
                    return model
                except:
                    pass
                
                # Coba load sebagai joblib
                try:
                    data = load(path)
                    print(f"✅ ANN model (joblib) loaded from: {path}")
                    print(f"   Type: {type(data)}")
                    
                    # Jika data adalah dictionary, cari model
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if hasattr(value, 'predict'):
                                print(f"   ✅ Found model in dict['{key}']")
                                return value
                        return data
                    
                    # Jika data adalah pipeline
                    elif hasattr(data, 'named_steps'):
                        for step_name, step_obj in data.named_steps.items():
                            if hasattr(step_obj, 'predict'):
                                print(f"   ✅ Found model in pipeline step: {step_name}")
                                return step_obj
                        return data
                    
                    return data
                except:
                    pass
        
        raise FileNotFoundError("No ANN model file found")
        
    except Exception as e:
        print(f"❌ Error loading ANN model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    """Load scaler"""
    try:
        possible_paths = [
            "model/scaler.joblib",
            "scaler.joblib",
            "model/ann_preprocessing_pipeline.joblib",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Coba load sebagai joblib
                try:
                    data = load(path)
                    print(f"✅ Scaler loaded from: {path}")
                    
                    # Jika data adalah pipeline, extract scaler
                    if hasattr(data, 'named_steps'):
                        for step_name, step_obj in data.named_steps.items():
                            if 'scaler' in step_name.lower() or hasattr(step_obj, 'transform'):
                                print(f"   Extracted scaler from pipeline step: {step_name}")
                                return step_obj
                    
                    # Jika data adalah dictionary, cari scaler
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if 'scaler' in key.lower() or hasattr(value, 'transform'):
                                print(f"   Found scaler in dict['{key}']")
                                return value
                    
                    # Jika data memiliki transform method
                    if hasattr(data, 'transform'):
                        return data
                    
                    return data
                except:
                    pass
        
        # Jika tidak ditemukan, coba extract dari pipeline
        if os.path.exists("app_pipeline.joblib"):
            try:
                pipeline = load("app_pipeline.joblib")
                if hasattr(pipeline, 'named_steps'):
                    for step_name, step_obj in pipeline.named_steps.items():
                        if 'scaler' in step_name.lower() or hasattr(step_obj, 'transform'):
                            print(f"✅ Scaler extracted from app_pipeline: {step_name}")
                            return step_obj
            except:
                pass
        
        raise FileNotFoundError("No scaler file found")
        
    except Exception as e:
        print(f"❌ Error loading scaler: {str(e)}")
        return None

@st.cache_resource
def load_encoder():
    """Load encoder"""
    try:
        possible_paths = [
            "model/encoder.joblib",
            "model/encode.joblib",
            "encoder.joblib",
            "encode.joblib",
            "model/ann_preprocessing_pipeline.joblib",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    data = load(path)
                    print(f"✅ Encoder loaded from: {path}")
                    
                    # Jika data adalah pipeline, extract encoder
                    if hasattr(data, 'named_steps'):
                        for step_name, step_obj in data.named_steps.items():
                            if 'encoder' in step_name.lower() or hasattr(step_obj, 'transform'):
                                print(f"   Extracted encoder from pipeline step: {step_name}")
                                return step_obj
                    
                    # Jika data adalah dictionary, cari encoder
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if 'encoder' in key.lower() or hasattr(value, 'transform'):
                                print(f"   Found encoder in dict['{key}']")
                                return value
                    
                    # Jika data memiliki transform method
                    if hasattr(data, 'transform'):
                        return data
                    
                    return data
                except:
                    pass
        
        # Jika tidak ditemukan, coba extract dari pipeline
        if os.path.exists("app_pipeline.joblib"):
            try:
                pipeline = load("app_pipeline.joblib")
                if hasattr(pipeline, 'named_steps'):
                    for step_name, step_obj in pipeline.named_steps.items():
                        if 'encoder' in step_name.lower() or hasattr(step_obj, 'transform'):
                            print(f"✅ Encoder extracted from app_pipeline: {step_name}")
                            return step_obj
            except:
                pass
        
        raise FileNotFoundError("No encoder file found")
        
    except Exception as e:
        print(f"❌ Error loading encoder: {str(e)}")
        return None

@st.cache_resource
def load_ridge_components():
    """Load all Ridge components at once"""
    try:
        model = load_ridge_model()
        scaler = load_scaler()
        encoder = load_encoder()
        return model, scaler, encoder
    except Exception as e:
        print(f"Error loading Ridge components: {str(e)}")
        return None, None, None

@st.cache_resource
def get_models():
    """Load all models at once"""
    try:
        model_ridge = load_ridge_model()
        model_ann = load_ann_model()
        scaler = load_scaler()
        encoder = load_encoder()
        return model_ridge, model_ann, scaler, encoder
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None

def predict_revenue_ridge(outlet, bulan, tahun, quarter=None, lag_1=0, lag_3=0, lag_6=0, rolling_3=0):
    """
    Predict revenue using Ridge model
    """
    try:
        model = load_ridge_model()
        scaler = load_scaler()
        encoder = load_encoder()
        
        print("="*50)
        print("DEBUG PREDIKSI RIDGE")
        print("="*50)
        print(f"Model type: {type(model)}")
        print(f"Model has predict: {hasattr(model, 'predict')}")
        
        if model is None:
            raise ValueError("Model failed to load")
        
        if not hasattr(model, 'predict'):
            raise ValueError(f"Model does not have 'predict' method. Type: {type(model)}")
        
        # Calculate quarter if not provided
        if quarter is None:
            quarter = (bulan - 1) // 3 + 1
        
        # Prepare input data
        input_dict = {
            'bulan': bulan,
            'tahun': tahun,
            'quarter': quarter,
            'lag_1': float(lag_1),
            'lag_3': float(lag_3),
            'lag_6': float(lag_6),
            'rolling_3': float(rolling_3),
            'Outlet': outlet
        }
        
        # Create DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # Encode outlet if encoder available
        if encoder is not None:
            try:
                outlet_encoded = encoder.transform(df_input[['Outlet']])
                outlet_cols = encoder.get_feature_names_out()
                outlet_df = pd.DataFrame(outlet_encoded, columns=outlet_cols)
                
                df_processed = pd.concat([
                    df_input[['bulan', 'tahun', 'quarter', 'lag_1', 'lag_3', 'lag_6', 'rolling_3']].reset_index(drop=True),
                    outlet_df
                ], axis=1)
            except Exception as e:
                print(f"Error encoding: {e}")
                df_processed = pd.get_dummies(df_input, columns=['Outlet'])
        else:
            df_processed = pd.get_dummies(df_input, columns=['Outlet'])
        
        # Feature columns
        feature_cols = [
            'bulan', 'tahun', 'quarter', 
            'lag_1', 'lag_3', 'lag_6', 'rolling_3',
            'Outlet_AYAM SERAYU - CABANG 1',
            'Outlet_AYAM SERAYU - CABANG 2',
            'Outlet_AYAM SERAYU - PUSAT'
        ]
        
        for col in feature_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        df_processed = df_processed[feature_cols]
        print(f"Processed shape: {df_processed.shape}")
        
        # Scale if scaler available
        if scaler is not None:
            try:
                X_scaled = scaler.transform(df_processed.values)
            except Exception as e:
                print(f"Error scaling: {e}")
                X_scaled = df_processed.values
        else:
            X_scaled = df_processed.values
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        print(f"Prediction: {prediction}")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def predict_revenue_ann(outlet, bulan, tahun, quarter=None, lag_1=0, lag_3=0, lag_6=0, rolling_3=0):
    """
    Predict revenue using ANN model
    """
    try:
        model = load_ann_model()
        scaler = load_scaler()
        encoder = load_encoder()
        
        print("="*50)
        print("DEBUG PREDIKSI ANN")
        print("="*50)
        print(f"Model type: {type(model)}")
        
        if model is None:
            raise ValueError("ANN model failed to load")
        
        # Calculate quarter if not provided
        if quarter is None:
            quarter = (bulan - 1) // 3 + 1
        
        # Prepare input data
        input_dict = {
            'bulan': bulan,
            'tahun': tahun,
            'quarter': quarter,
            'lag_1': float(lag_1),
            'lag_3': float(lag_3),
            'lag_6': float(lag_6),
            'rolling_3': float(rolling_3),
            'Outlet': outlet
        }
        
        # Create DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # Encode outlet if encoder available
        if encoder is not None:
            try:
                outlet_encoded = encoder.transform(df_input[['Outlet']])
                outlet_cols = encoder.get_feature_names_out()
                outlet_df = pd.DataFrame(outlet_encoded, columns=outlet_cols)
                
                df_processed = pd.concat([
                    df_input[['bulan', 'tahun', 'quarter', 'lag_1', 'lag_3', 'lag_6', 'rolling_3']].reset_index(drop=True),
                    outlet_df
                ], axis=1)
            except Exception as e:
                print(f"Error encoding: {e}")
                df_processed = pd.get_dummies(df_input, columns=['Outlet'])
        else:
            df_processed = pd.get_dummies(df_input, columns=['Outlet'])
        
        # Feature columns
        feature_cols = [
            'bulan', 'tahun', 'quarter', 
            'lag_1', 'lag_3', 'lag_6', 'rolling_3',
            'Outlet_AYAM SERAYU - CABANG 1',
            'Outlet_AYAM SERAYU - CABANG 2',
            'Outlet_AYAM SERAYU - PUSAT'
        ]
        
        for col in feature_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        df_processed = df_processed[feature_cols]
        
        # Scale if scaler available
        if scaler is not None:
            try:
                X_scaled = scaler.transform(df_processed.values)
            except Exception as e:
                print(f"Error scaling: {e}")
                X_scaled = df_processed.values
        else:
            X_scaled = df_processed.values
        
        # Predict
        if hasattr(model, 'predict'):
            result = model.predict(X_scaled)
            if isinstance(result, list) or isinstance(result, np.ndarray):
                prediction = result[0][0] if len(result[0]) > 0 else result[0]
            else:
                prediction = result
        else:
            prediction = model.predict(X_scaled)[0]
        
        print(f"Prediction: {prediction}")
        return prediction
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

# Untuk kompatibilitas dengan kode lama
def predict_revenue(outlet, bulan, tahun, quarter=None, lag_1=0, lag_3=0, lag_6=0, rolling_3=0):
    """Wrapper function for backward compatibility"""
    return predict_revenue_ridge(outlet, bulan, tahun, quarter, lag_1, lag_3, lag_6, rolling_3)

# Test function
def test_models():
    """Test all models"""
    print("\n" + "="*50)
    print("TESTING MODELS")
    print("="*50)
    
    ridge = load_ridge_model()
    ann = load_ann_model()
    scaler = load_scaler()
    encoder = load_encoder()
    
    print(f"Ridge model: {ridge is not None} - Type: {type(ridge)}")
    print(f"ANN model: {ann is not None} - Type: {type(ann)}")
    print(f"Scaler: {scaler is not None} - Type: {type(scaler)}")
    print(f"Encoder: {encoder is not None} - Type: {type(encoder)}")
    
    if ridge is not None and hasattr(ridge, 'predict'):
        print("✅ Ridge model is ready")
    else:
        print("❌ Ridge model not ready")
    
    return ridge, ann, scaler, encoder

if __name__ == "__main__":
    test_models()