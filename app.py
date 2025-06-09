import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("svm_model.pkl")

# Rata-rata dan standar deviasi dari data training (untuk standardisasi)
# (Asumsi berdasarkan dataset, sesuaikan dengan data pelatihan asli jika berbeda)
means = {'Age': 23.144453, 'Height': 1.715131, 'Weight': 92.504880, 'FCVC': 2.456863,
         'CH2O': 2.060705, 'FAF': 1.031469, 'TUE': 0.693352, 'NCP': 2.759646, 'BMI': 31.389988}
stds = {'Age': 4.102400, 'Height': 0.087400, 'Weight': 27.473986, 'FCVC': 0.546924,
        'CH2O': 0.601654, 'FAF': 0.855703, 'TUE': 0.586877, 'NCP': 0.753488, 'BMI': 7.123892}

def standardize(value, mean, std):
    return (value - mean) / std

def tampilkan_hasil_prediksi(label_prediksi):
    info = {
        "Insufficient_Weight": {
            "desc": "Anda termasuk dalam kategori Berat Badan Kurang. Ini berarti tubuh Anda mungkin memerlukan asupan gizi yang lebih untuk mencapai berat badan sehat.",
            "rekomendasi": "Disarankan untuk berkonsultasi dengan ahli gizi untuk pola makan yang sesuai dan menjaga kesehatan secara keseluruhan.",
            "color": "blue"
        },
        "Normal_Weight": {
            "desc": "Berat badan Anda berada pada kisaran normal yang sehat. Pertahankan pola hidup aktif dan konsumsi makanan bergizi seimbang.",
            "rekomendasi": "Lanjutkan gaya hidup sehat dan rutin cek kesehatan secara berkala.",
            "color": "green"
        },
        "Overweight_Level_I": {
            "desc": "Anda masuk dalam kategori Kelebihan Berat Badan Tingkat I. Ini adalah peringatan awal untuk mulai memperhatikan pola makan dan aktivitas fisik.",
            "rekomendasi": "Disarankan untuk meningkatkan aktivitas fisik dan mengurangi konsumsi makanan tinggi kalori secara bertahap.",
            "color": "yellow"
        },
        "Overweight_Level_II": {
            "desc": "Kategori Kelebihan Berat Badan Tingkat II. Risiko masalah kesehatan mulai meningkat jika tidak ada perubahan gaya hidup.",
            "rekomendasi": "Segera konsultasikan dengan profesional kesehatan dan buatlah rencana diet serta olahraga yang terstruktur.",
            "color": "orange"
        },
        "Obesity_Type_I": {
            "desc": "Anda termasuk Obesitas Tipe I, yang berarti ada penumpukan lemak berlebih yang dapat meningkatkan risiko penyakit kronis.",
            "rekomendasi": "Konsultasi dengan dokter atau ahli gizi sangat dianjurkan untuk memulai program penurunan berat badan yang aman dan efektif.",
            "color": "orange"
        },
        "Obesity_Type_II": {
            "desc": "Obesitas Tipe II, kondisi ini sudah termasuk tingkat berat dengan risiko kesehatan yang signifikan.",
            "rekomendasi": "Perubahan gaya hidup dan pengawasan medis yang ketat diperlukan untuk menghindari komplikasi serius.",
            "color": "red"
        },
        "Obesity_Type_III": {
            "desc": "Obesitas Tipe III (Obesitas Morbid) sangat serius dan memerlukan intervensi medis segera.",
            "rekomendasi": "Segera lakukan konsultasi dengan dokter spesialis untuk penanganan yang tepat, bisa meliputi terapi medis atau operasi jika diperlukan.",
            "color": "red"
        }
    }

    hasil = info.get(label_prediksi, None)
    if hasil:
        st.markdown(f"<h3 style='color:{hasil['color']};'>Hasil Prediksi: {label_prediksi.replace('_', ' ')}</h3>", unsafe_allow_html=True)
        st.write(hasil['desc'])
        st.markdown(f"*Rekomendasi:* {hasil['rekomendasi']}")
    else:
        st.error("Terjadi kesalahan dalam prediksi. Silakan coba lagi.")

label_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

st.title("Prediksi Kategori Obesitas")

# Input
age = st.number_input("Usia (tahun)", min_value=10, max_value=100)
if age > 35 or age < 14:
    st.warning("Model ini dilatih pada data usia 14 hingga 61 tahun. Hasil prediksi untuk usia di luar rentang ini mungkin tidak akurat.")

gender = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")

height = st.number_input("Tinggi Badan (meter)", min_value=1.20, max_value=2.20, step=0.01)
if height < 1.45 or height > 1.98:
    st.info("Model dilatih dengan tinggi badan antara 1.45 m hingga 1.99 m. Prediksi di luar rentang ini mungkin kurang tepat.")

weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0)
if weight < 39.0 or weight > 173.0:
    st.info("Data pelatihan mencakup berat antara 39 kg hingga 174 kg. Di luar rentang ini, prediksi bisa kurang akurat.")

# Hitung BMI otomatis dan tampilkan sebagai input
bmi_calculated = weight / (height ** 2)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=bmi_calculated, step=0.1)
if abs(bmi - bmi_calculated) > 0.1:
    st.warning("BMI yang Anda masukkan tidak sesuai dengan perhitungan otomatis (Berat / Tinggi^2). Pertimbangkan untuk menggunakan nilai otomatis.")

favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

fcvc = st.number_input("Frekuensi Konsumsi Sayur (1 - 3)", min_value=1.0, max_value=3.0, step=0.1)

ncp = st.number_input("Jumlah Makan Utama (1 - 4)", min_value=1.0, max_value=4.0, step=0.1)

scc = st.selectbox("Konsultasi Gizi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

smoke = st.selectbox("Apakah Merokok?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

ch2o = st.number_input("Konsumsi Air Harian (1 - 3 liter)", min_value=1.0, max_value=4.0, step=0.1)
if ch2o < 1.0 or ch2o > 3.0:
    st.info("Model dilatih dengan konsumsi air antara 1 hingga 3 liter. Nilai di luar rentang ini dapat memengaruhi akurasi prediksi.")

family_history = st.selectbox("Riwayat Keluarga Obesitas", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

faf = st.number_input("Aktivitas Fisik Mingguan (jam)", min_value=0.0, max_value=6.0, step=0.1)
if faf < 0.0 or faf > 3.0:
    st.info("Aktivitas fisik dalam data pelatihan berkisar 0 hingga 3 jam. Di atas itu, akurasi model bisa berkurang.")

tue = st.number_input("Waktu di Depan Layar per Hari (jam)", min_value=0.0, max_value=10.0, step=0.1)
if tue < 0.0 or tue > 2.0:
    st.info("Model dilatih dengan waktu layar harian antara 0 hingga 1 jam 45 menit. Nilai lebih dari itu dapat menghasilkan prediksi yang kurang akurat.")

# Input untuk CAEC (one-hot encoding)
caec = st.selectbox("Frekuensi Makan Berlebihan", ["Tidak", "Sering", "Selalu"], index=0)
caec_no = 1.0 if caec == "Tidak" else 0.0
caec_frequently = 1.0 if caec == "Sering" else 0.0
caec_always = 1.0 if caec == "Selalu" else 0.0

# Input untuk CALC (one-hot encoding)
calc = st.selectbox("Konsumsi Alkohol", ["Tidak", "Sering", "Selalu"], index=0)
calc_no = 1.0 if calc == "Tidak" else 0.0
calc_frequently = 1.0 if calc == "Sering" else 0.0
calc_always = 1.0 if calc == "Selalu" else 0.0

# Input untuk MTRANS (one-hot encoding)
mtrans = st.selectbox("Transportasi Harian", ["Transportasi Publik", "Berjalan Kaki", "Mobil", "Motor", "Sepeda"], index=0)
mtrans_public = 1.0 if mtrans == "Transportasi Publik" else 0.0
mtrans_walking = 1.0 if mtrans == "Berjalan Kaki" else 0.0
mtrans_automobile = 1.0 if mtrans == "Mobil" else 0.0
mtrans_motorbike = 1.0 if mtrans == "Motor" else 0.0
mtrans_bike = 1.0 if mtrans == "Sepeda" else 0.0

# Hitung BMI
bmi = weight / (height ** 2)

if st.button("Prediksi"):
    # Siapkan input terstandardisasi dengan 26 fitur
    X_input = [
        gender,                                                # 1. Gender
        standardize(age, means['Age'], stds['Age']),           # 2. Age
        standardize(height, means['Height'], stds['Height']),  # 3. Height
        standardize(weight, means['Weight'], stds['Weight']),  # 4. Weight
        favc,                                                  # 5. FAVC
        standardize(fcvc, means['FCVC'], stds['FCVC']),        # 6. FCVC
        standardize(ncp, means['NCP'], stds['NCP']),           # 7. NCP
        scc,                                                   # 8. SCC
        smoke,                                                 # 9. SMOKE
        standardize(ch2o, means['CH2O'], stds['CH2O']),        # 10. CH2O
        family_history,                                        # 11. family_history_with_overweight
        standardize(faf, means['FAF'], stds['FAF']),           # 12. FAF
        standardize(tue, means['TUE'], stds['TUE']),           # 13. TUE
        caec_no,                                               # 14. CAEC_no
        caec_frequently,                                       # 15. CAEC_Frequently
        caec_always,                                           # 16. CAEC_Always
        calc_no,                                               # 17. CALC_no
        calc_frequently,                                       # 18. CALC_Frequently
        calc_always,                                           # 19. CALC_Always
        mtrans_public,                                         # 20. MTRANS_Public_Transportation
        mtrans_walking,                                        # 21. MTRANS_Walking
        mtrans_automobile,                                     # 22. MTRANS_Automobile
        mtrans_motorbike,                                      # 23. MTRANS_Motorbike
        mtrans_bike,                                           # 24. MTRANS_Bike
        standardize(bmi, means['BMI'], stds['BMI']),           # 25. BMI
        0.0                                                    # 26. Placeholder (fitur tambahan lain, jika ada)
    ]

    pred = model.predict([X_input])[0]
    label_prediksi = label_map[pred]

    tampilkan_hasil_prediksi(label_prediksi)