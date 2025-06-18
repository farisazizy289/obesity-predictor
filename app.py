import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Prediksi Obesitas",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Kustom untuk Tampilan UI yang Lebih Baik ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1F485D 0%, #D8E0A5 100%); /* Gradasi dari ocean ke lime */
        color: #ffffff; /* Putih untuk teks agar kontras dengan latar */
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        color: #D8E0A5; /* Lime untuk judul utama */
        text-align: center;
        font-size: 3em;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        color: #ffffff; /* Putih untuk subjudul */
        font-size: 2em;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #D8E0A5; /* Lime untuk border */
    }
    h3 {
        color: #ffffff; /* Putih untuk subjudul kecil */
        text-align: center;
        font-size: 1.5em;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stForm {
        background: ##1B2A2A ; 
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
    }
    .stSelectbox > label, .stNumberInput > label {
        font-size: 1.1em;
        font-weight: semibold;
        color: #ffffff; /* Putih */
        margin-bottom: 8px;
    }
    .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] {
        font-size: 1.1em;
        font-weight: semibold;
        color: #ffffff; /* Putih */
        margin-bottom: 8px;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stNumberInput div[data-baseweb="input"]:hover {
        border-color: #D8E0A5; /* Lime pada hover */
        background-color: #1F485D; /* Ocean pada hover */
        color: #ffffff; /* Putih untuk teks saat hover */
    }
    .stSelectbox div[data-baseweb="select"] input, .stNumberInput div[data-baseweb="input"] input {
        color: #1F485D !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1F485D 0%, #D8E0A5 100%); /* Gradasi ocean ke lime */
        color: #ffffff;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1.2em;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #173a4a 0%, #c1d08d 100%); /* Variasi lebih gelap dan terang */
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .stDataFrame table {
        background-color: #D8E0A5; /* Lime */
        color: #1F485D; /* Ocean */
    }
    .stDataFrame th {
        background-color: #1F485D; /* Ocean */
        color: #ffffff;
        font-weight: bold;
    }
    .stDataFrame td {
        border-top: 1px solid #1F485D; /* Ocean untuk border */
    }
    .stSuccess, .stError, .stWarning {
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #ffffff;
        font-weight: bold;
    }
    .stSuccess {
        background: linear-gradient(90deg, #1F485D 0%, #D8E0A5 100%); /* Gradasi ocean ke lime */
    }
    .stError {
        background-color: #ff6666;
    }
    .stWarning {
        background-color: #ffd700;
        color: #2c3e50;
    }
    .css-1offfwp {
        padding: 0 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    try:
        all_tuned_models = joblib.load('models/all_tuned_models.pkl')
        model = all_tuned_models['SVM']
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = label_encoders['NObeyesdad']
        return model, scaler, label_encoders, target_encoder
    except Exception as e:
        st.error(f"Gagal memuat model/preprocessing: {e}")
        st.stop()

def preprocess_input(input_df, label_encoders, scaler):
    categorical_cols = [col for col in label_encoders.keys() if col != 'NObeyesdad']
    
    for col in categorical_cols:
        if col in input_df.columns:
            if input_df[col].iloc[0] not in label_encoders[col].classes_:
                st.warning(f"Nilai tidak dikenal di kolom {col}.")
                input_df[col] = pd.Categorical(input_df[col], categories=label_encoders[col].classes_).codes[0]
            else:
                input_df[col] = label_encoders[col].transform(input_df[col])

    if hasattr(scaler, 'feature_names_in_'):
        input_df = input_df[scaler.feature_names_in_]
    
    try:
        return scaler.transform(input_df)
    except Exception as e:
        st.error(f"Gagal preprocessing data: {e}")
        st.stop()

# Validasi helper
def to_float(val, field):
    try:
        return float(val)
    except:
        st.error(f"Input '{field}' harus berupa angka.")
        st.stop()

def main():
    st.markdown("<h1 class='stTitle'>Prediksi Tingkat Obesitas</h1>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("<h3 class='stTitle'>Input Data Pasien </h3>", unsafe_allow_html=True)

        gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        age = to_float(st.text_input("Usia (tahun)", "25"), "Usia")
        height = to_float(st.text_input("Tinggi (meter)", "1.80"), "Tinggi")
        weight = to_float(st.text_input("Berat (kg)", "85.0"), "Berat")
        family_history = st.selectbox("Riwayat Keluarga Obesitas", ['yes', 'no'])
        favc = st.selectbox("Sering Makan Tinggi Kalori", ['yes', 'no'])
        fcvc = to_float(st.text_input("Frekuensi Konsumsi Sayuran (1-3)", "2.0"), "FCVC")
        ncp = to_float(st.text_input("Jumlah Makan Utama (1-4)", "3.0"), "NCP")
        caec = st.selectbox("Konsumsi Antar Waktu Makan", ['no', 'Sometimes', 'Frequently', 'Always'])
        smoke = st.selectbox("Merokok", ['yes', 'no'])
        ch2o = to_float(st.text_input("Konsumsi Air (liter)", "2.0"), "CH2O")
        scc = st.selectbox("Monitor Konsumsi Kalori", ['yes', 'no'])
        faf = to_float(st.text_input("Frekuensi Aktivitas Fisik", "1.0"), "FAF")
        tue = to_float(st.text_input("Waktu Teknologi (jam)", "1.0"), "TUE")
        calc = st.selectbox("Konsumsi Alkohol", ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox("Transportasi Utama", ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_data = {
            'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history_with_overweight': [family_history], 'FAVC': [favc], 'FCVC': [fcvc],
            'NCP': [ncp], 'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc],
            'FAF': [faf], 'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
        }
        input_df = pd.DataFrame(input_data)

        model, scaler, label_encoders, target_encoder = load_models()
        input_scaled = preprocess_input(input_df, label_encoders, scaler)

        if input_scaled is not None:
            prediction_encoded = model.predict(input_scaled)
            prediction_label = target_encoder.inverse_transform(prediction_encoded)
            st.markdown(f"<div class='stSuccess'>Hasil Prediksi: Tingkat Obesitas Anda adalah <b>{prediction_label[0]}</b></div>", unsafe_allow_html=True)
            st.write("Data Input Anda:", input_df)

if __name__ == "__main__":
    main()