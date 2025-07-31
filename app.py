import streamlit as st
import openai
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf

# Set OpenAI API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🫀 PCG Waveform Analyzer with ChatGPT")
st.markdown("Upload a PCG (Phonocardiogram) `.wav` file to analyze and get AI-powered diagnosis.")

# Upload audio file
uploaded_file = st.file_uploader("Upload your PCG WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Load audio
    signal, sr = librosa.load(uploaded_file, sr=None)
    
    # Show waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title("PCG Waveform")
    st.pyplot(fig)

    # Extract features
    duration = librosa.get_duration(y=signal, sr=sr)
    rms = np.mean(librosa.feature.rms(y=signal))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    
    st.markdown("### 📊 Extracted Audio Features")
    st.write(f"**Duration**: {duration:.2f} sec")
    st.write(f"**RMS Energy**: {rms:.6f}")
    st.write(f"**Zero Crossing Rate**: {zcr:.6f}")
    st.write(f"**Spectral Centroid**: {centroid:.2f}")

    # Convert features to prompt
    prompt = f"""
    I have uploaded a heart sound recording. Based on the features extracted:
    - Duration: {duration:.2f} sec
    - RMS Energy: {rms:.6f}
    - Zero Crossing Rate: {zcr:.6f}
    - Spectral Centroid: {centroid:.2f} Hz
    
    Please analyze the above features and suggest if this PCG is:
    1. Normal
    2. Suggestive of any murmur (e.g., mitral stenosis, aortic regurgitation, etc.)
    3. Any warning signs or further tests needed
    
    Please explain in detail like a clinical expert.
    """

    if st.button("🧠 Analyze with ChatGPT"):
        with st.spinner("Asking ChatGPT for diagnosis..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a medical expert in cardiac auscultation and phonocardiography."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=700
                )
                diagnosis = response['choices'][0]['message']['content']
                st.success("✅ ChatGPT Diagnosis:")
                st.markdown(diagnosis)
            except Exception as e:
                st.error(f"Error while contacting OpenAI: {str(e)}")
