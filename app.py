import streamlit as st
import numpy as np
import joblib


mean_values = {
    'age': 24.61197449498307,
    'gender': 1.5011035634512278,
    'height': 1.709719796171931,
    'weight': 86.79328512552229,
    'favc': 1.8777203493488603,
    'fcvc': 2.429486097245775,
    'ncp': 2.754221343222809,
    'scc': 1.0557468924253193,
    'smoke': 1.0209743482819817,
    'ch2o': 2.005106663319832,
    'family_history': 1.8094146841052146,
    'faf': 1.1021826251820754,
    'tue': 0.6859587423237127,
    'caec_always': 0.02467904688043662,
    'caec_frequently': 0.11796812177879165,
    'caec_sometimes': 0.8314324754306504,
    'caec_no': 0.018276072847472555,
    'calc_always': 0.0005089058524173028,
    'calc_frequently': 0.03540504107997242,
    'calc_sometimes': 0.6545487767994426,
    'calc_no': 0.301612871820779,
    'mtrans_automobile': 0.20697574046210301,
    'mtrans_bike': 0.0021520601351102406,
    'mtrans_motorbike': 0.006367899292524453,
    'mtrans_public': 0.7504304676868959,
    'mtrans_walking': 0.026379102161869554
}

std_values = {
    'age': 8.286430915782583,
    'gender': 0.514787430617852,
    'height': 0.12048799374271067,
    'weight': 32.77243110396893,
    'favc': 0.3554183321599801,
    'fcvc': 0.6296023430061097,
    'ncp': 0.9152150884847503,
    'scc': 0.2605590225979267,
    'smoke': 0.17573766028623242,
    'ch2o': 0.6794613891049351,
    'family_history': 0.41126637753578055,
    'faf': 1.1403906499334708,
    'tue': 0.6846236192898081,
    'caec_always': 0.1527060253941316,
    'caec_frequently': 0.3190617516319021,
    'caec_sometimes': 0.37019040714155027,
    'caec_no': 0.1313511309328874,
    'calc_always': 0.022558941739747384,
    'calc_frequently': 0.18121957174445225,
    'calc_sometimes': 0.4712990507884281,
    'calc_no': 0.4553335577019287,
    'mtrans_automobile': 0.40209645603205446,
    'mtrans_bike': 0.045372681543394695,
    'mtrans_motorbike': 0.0775871710105017,
    'mtrans_public': 0.4285289098556206,
    'mtrans_walking': 0.1541490163354169
}


def manual_standardization(X, mean_values, std_values):
    X_scaled = []
    for i, key in enumerate(mean_values.keys()):
        X_scaled.append((X[i] - mean_values[key]) / std_values[key] if std_values[key] != 0 else X[i])
    return np.array(X_scaled).reshape(1, -1)


model = joblib.load("rf_model_best.pkl")  

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
    2: "Obesity_Type_I",
    3: "Obesity_Type_II",
    4: "Obesity_Type_III",
    5: "Overweight_Level_I",
    6: "Overweight_Level_II"
}

st.title("Prediksi Kategori Obesitas")

# Input
age = st.number_input("Usia (tahun)", min_value=10, max_value=100)
if age > 14 or age < 106:
    st.warning("Catatan: Model dilatih pada data dengan usia minimal 14 tahun. Usia di atas 100 tahun bersifat outlier dan mungkin mempengaruhi akurasi.")

gender = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")

height = st.number_input("Tinggi Badan (meter)", min_value=1.20, max_value=2.20, step=0.01)
if height < 1.45 or height > 2.85:
    st.info("Catatan: Nilai tinggi lebih dari 2 meter jarang terjadi, kemungkinan merupakan data ekstrem atau tidak valid.")

weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0)
if weight < 39.0 or weight > 173.0:
    st.info("Catatan: Berat badan lebih dari 200 kg kemungkinan besar merupakan data ekstrem atau kasus klinis khusus.")

favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori? (FAVC)", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

fcvc = st.number_input("Frekuensi Konsumsi Sayur (1 - 3) (FCVC)", min_value=1.0, max_value=3.0, step=0.1)

ncp = st.number_input("Jumlah Makan Utama (1 - 4) (NCP)", min_value=1.0, max_value=4.0, step=0.1)

scc = st.selectbox("Konsultasi Gizi? (SCC)", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

smoke = st.selectbox("Apakah Merokok? (SMOKE)", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

ch2o = st.number_input("Konsumsi Air Harian (1 - 3 liter) (CH2O)", min_value=1.0, max_value=4.0, step=0.1)
if ch2o < 1.0 or ch2o > 9.23:
    st.info("Catatan: Konsumsi lebih dari 5 liter per hari jarang terjadi, kemungkinan merupakan kasus ekstrem.")

family_history = st.selectbox("Riwayat Keluarga Obesitas", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

faf = st.number_input("Aktivitas Fisik Mingguan (jam) (FAF)", min_value=0.0, max_value=6.0, step=0.1)
if faf < 0.0 or faf > 3.0:
    st.info("Aktivitas fisik dalam data pelatihan berkisar 0 hingga 12 jam 16 menit. Di atas itu, akurasi model bisa berkurang.")

tue = st.number_input("Waktu di Depan Layar per Hari (jam) (TUE)", min_value=0.0, max_value=10.0, step=0.1)
if tue < 0.0 or tue > 7.7:
    st.info("Model dilatih dengan waktu layar harian antara 0 hingga 7 jam 40 menit. Nilai lebih dari itu dapat menghasilkan prediksi yang kurang akurat.")

# Input CAEC (Frekuensi Makan Berlebihan)
caec = st.selectbox("Frekuensi Makan Berlebihan (CAEC)", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
caec_no = 1.0 if caec == "Tidak" else 0.0
caec_sometimes = 1.0 if caec == "Kadang-kadang" else 0.0
caec_frequently = 1.0 if caec == "Sering" else 0.0
caec_always = 1.0 if caec == "Selalu" else 0.0

# Input CALC (Konsumsi Alkohol)
calc = st.selectbox("Konsumsi Alkohol (CALC)", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
calc_no = 1.0 if calc == "Tidak" else 0.0
calc_sometimes = 1.0 if calc == "Kadang-kadang" else 0.0
calc_frequently = 1.0 if calc == "Sering" else 0.0
calc_always = 1.0 if calc == "Selalu" else 0.0

# Input MTRANS (Transportasi)
mtrans = st.selectbox("Transportasi Harian (MTRANS)", ["Transportasi Publik", "Berjalan Kaki", "Mobil", "Motor", "Sepeda"])
mtrans_public = 1.0 if mtrans == "Transportasi Publik" else 0.0
mtrans_walking = 1.0 if mtrans == "Berjalan Kaki" else 0.0
mtrans_automobile = 1.0 if mtrans == "Mobil" else 0.0
mtrans_motorbike = 1.0 if mtrans == "Motor" else 0.0
mtrans_bike = 1.0 if mtrans == "Sepeda" else 0.0

# BMI
bmi = weight / (height ** 2)

if st.button("Prediksi"):
    # Susun input sesuai urutan pelatihan
    X_input = [
        age, gender, height, weight, favc, fcvc, ncp, scc, smoke, ch2o, family_history, faf, tue,
        caec_always, caec_frequently, caec_sometimes, caec_no,
        calc_always, calc_frequently, calc_sometimes, calc_no,
        mtrans_automobile, mtrans_bike, mtrans_motorbike, mtrans_public, mtrans_walking
    ]

    # Terapkan standarisasi manual
    X_scaled = manual_standardization(X_input, mean_values, std_values)
    
    # Prediksi dengan Random Forest
    pred = model.predict(X_scaled)[0]
    label_prediksi = label_map[pred]
    tampilkan_hasil_prediksi(label_prediksi)