import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import os
import google.generativeai as genai
from fpdf import FPDF
import base64

# === CONFIGURE GEMINI ===
genai.configure(api_key="AIzaSyCb5kK8XnscVVfE0wItNT0c-hZbWyqzFaA")
model = genai.GenerativeModel("gemini-pro")

# === STREAMLIT UI ===
st.set_page_config(page_title="PCG Analyzer with Gemini", layout="centered")
st.title("🫀 PCG Analyzer using Gemini AI")
st.markdown("Upload a **.wav** phonocardiogram (PCG) file to analyze heart sound abnormalities.")

# === UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload PCG File (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.success("✅ File uploaded successfully!")

    # === TEMP SAVE ===
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.write(uploaded_file.read())
    temp_wav.close()

    # === LOAD AUDIO ===
    y, sr = librosa.load(temp_wav.name, sr=None)
    duration = len(y) / sr
    st.write(f"📏 Duration: `{duration:.2f}` seconds")
    st.write(f"🔊 Sample Rate: `{sr}` Hz")

    # === WAVEFORM ===
    st.subheader("📉 Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # === SPECTROGRAM ===
    st.subheader("🎞️ Spectrogram")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set_title("Log-frequency power spectrogram")
    st.pyplot(fig2)

    # === MFCC FEATURES ===
    st.subheader("🎛️ Extracted MFCC Features")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    for i, val in enumerate(mfcc_mean):
        st.write(f"MFCC {i+1}: {val:.2f}")

    # === GEMINI DIAGNOSIS ===
    st.subheader("🧠 Gemini AI Diagnostic Suggestion")
    prompt = f"""
    The following are mean MFCC features extracted from a phonocardiogram (PCG) recording:

    {mfcc_mean.tolist()}

    Analyze these features and determine whether this suggests any cardiac abnormalities such as murmurs, valve stenosis, regurgitation, or arrhythmia.
    Provide a clear, clinical-style interpretation, including a possible diagnosis and why.
    """

    with st.spinner("Analyzing with Gemini AI..."):
        try:
            response = model.generate_content(prompt)
            diagnosis = response.text
            st.success("📋 AI Diagnosis:")
            st.markdown(diagnosis)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
            diagnosis = "Error during diagnosis."

    # === PDF REPORT DOWNLOAD ===
    st.subheader("📄 Download Diagnosis Report")

    def create_pdf(text, filename="pcg_diagnosis.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "PCG AI Diagnosis Report", align='C')
        pdf.ln()
        pdf.multi_cell(0, 10, f"Diagnosis:\n\n{text}")
        pdf.output(filename)

    pdf_file = "pcg_diagnosis.pdf"
    create_pdf(diagnosis, pdf_file)

    with open(pdf_file, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="pcg_diagnosis.pdf">📥 Download Report as PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    os.remove(temp_wav.name)
