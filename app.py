import streamlit as st
import openai
import librosa
import numpy as np
import io

# Set your OpenAI API key
openai.api_key = "sk-...your-valid-key..."

st.set_page_config(page_title="PCG Waveform Analyzer", layout="centered")
st.title("🫀 PCG Waveform Analyzer with ChatGPT")

# Upload .wav file
uploaded_file = st.file_uploader("Upload a PCG (.wav) file", type=["wav"])

def extract_features(audio, sr):
    duration = librosa.get_duration(y=audio, sr=sr)
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return duration, rms, zcr

def get_chatgpt_diagnosis(prompt_summary):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a medical expert in analyzing heart sound waveforms (PCG)."},
                {"role": "user", "content": prompt_summary}
            ]
        )
        diagnosis = response["choices"][0]["message"]["content"]
        return diagnosis
    except Exception as e:
        st.error(f"❌ Error during ChatGPT analysis: {e}")
        return None

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    try:
        # Load audio
        audio, sr = librosa.load(uploaded_file, sr=None)
        duration, rms, zcr = extract_features(audio, sr)

        st.success("✅ Audio loaded and features extracted!")
        st.write(f"📏 Duration: {duration:.2f} sec")
        st.write(f"📈 RMS Energy: {rms:.4f}")
        st.write(f"🔀 Zero Crossing Rate: {zcr:.4f}")

        # Prepare prompt
        prompt = (
            f"I have recorded a heart sound waveform of duration {duration:.2f} seconds. "
            f"The RMS energy is {rms:.4f} and the zero-crossing rate is {zcr:.4f}. "
            f"Based on this waveform summary, could you provide a possible cardiac condition (e.g., murmur, stenosis, regurgitation)?"
        )

        if st.button("🔍 Analyze with ChatGPT"):
            diagnosis = get_chatgpt_diagnosis(prompt)
            if diagnosis:
                st.subheader("🧠 ChatGPT Diagnosis:")
                st.write(diagnosis)

    except Exception as e:
        st.error(f"❌ Error loading the file: {e}")
