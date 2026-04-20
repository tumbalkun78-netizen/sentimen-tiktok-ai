import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Analisis Sentimen TikTok AI",
    page_icon="📱",
    layout="centered"
)

# =========================
# LOAD MODEL & TOKENIZER
# =========================
model = load_model("model_sentimen_lstm.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# CLEANING FUNCTION
# =========================


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


label_map = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

emoji_map = {
    "Negatif": "❌",
    "Netral": "😐",
    "Positif": "✅"
}

color_map = {
    "Negatif": "#ff4b4b",
    "Netral": "#f0ad4e",
    "Positif": "#28a745"
}

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 850px;
    }

    .judul-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 28px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        margin-bottom: 20px;
    }

    .judul {
        font-size: 38px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
        text-align: center;
    }

    .subjudul {
        font-size: 17px;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 0;
    }

    .section-box {
        background: rgba(255,255,255,0.04);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-top: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .hasil-box {
        padding: 20px;
        border-radius: 18px;
        margin-top: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    .hasil-title {
        font-size: 18px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    .hasil-label {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hasil-clean {
        font-size: 15px;
        color: #e5e7eb;
        background: rgba(255,255,255,0.06);
        padding: 12px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .mini-note {
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 8px;
    }

    .contoh-box {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 10px;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.06);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 📌 Informasi Model")
st.sidebar.write("**Metode:** LSTM")
st.sidebar.write("**Akurasi:** 88.10%")
st.sidebar.write("**Dataset:** Komentar TikTok tentang fitur AI")
st.sidebar.write("**Kelas Sentimen:** Negatif, Netral, Positif")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="judul-box">
    <div class="judul">📱 Analisis Sentimen TikTok AI</div>
    <div class="subjudul">Aplikasi prediksi sentimen komentar netizen terhadap fitur AI pada TikTok menggunakan model LSTM</div>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT AREA
# =========================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("### ✍️ Masukkan Komentar")
komentar = st.text_area(
    "",
    height=160,
    placeholder="Contoh: fiturnya keren banget, tapi di hp aku malah ngelag"
)
st.markdown('<div class="mini-note">Masukkan satu komentar, lalu klik tombol prediksi.</div>',
            unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    prediksi = st.button("🔍 Prediksi Sentimen", use_container_width=True)

with col2:
    if st.button("🧹 Bersihkan Input", use_container_width=True):
        st.rerun()

# =========================
# CONTOH KOMENTAR
# =========================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("### 💬 Contoh Komentar")
st.markdown("""
<div class="contoh-box">✅ "fiturnya keren banget dan membantu bikin konten"</div>
<div class="contoh-box">❌ "fiturnya bikin hp ngelag dan sering hilang"</div>
<div class="contoh-box">😐 "manfaatnya apa dan cara pakainya gimana"</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if prediksi:
    if komentar.strip() == "":
        st.warning("Komentar tidak boleh kosong.")
    else:
        teks_bersih = clean_text(komentar)
        seq = tokenizer.texts_to_sequences([teks_bersih])
        padded = pad_sequences(seq, maxlen=50)

        pred = model.predict(padded, verbose=0)
        hasil_index = np.argmax(pred, axis=1)[0]
        hasil_label = label_map[hasil_index]
        hasil_emoji = emoji_map[hasil_label]
        hasil_color = color_map[hasil_label]
        confidence = float(np.max(pred) * 100)

        st.markdown(
            f"""
            <div class="hasil-box" style="background: linear-gradient(135deg, {hasil_color}22, rgba(255,255,255,0.04));">
                <div class="hasil-title">📊 Hasil Prediksi</div>
                <div class="hasil-label" style="color:{hasil_color};">
                    {hasil_emoji} {hasil_label}
                </div>
                <div style="color:white; font-size:16px;">
                    Tingkat keyakinan model: <b>{confidence:.2f}%</b>
                </div>
                <div class="hasil-clean">
                    <b>Hasil preprocessing:</b><br>{teks_bersih}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
