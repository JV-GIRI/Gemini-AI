import streamlit as st
import openai
import librosa
import numpy as np
import tempfile

# Initialize OpenAI client (requires openai>=1.0.0)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="PCG Audio Analyzer with ChatGPT", layout="centered")
st.title("🫀 PCG (.wav) Audio Analyzer with ChatGPT")
st.write("Upload a PCG heart sound (.wav) file to get a diagnostic interpretation.")

# Upload PCG .wav file
uploaded_file = st.file_uploader("Upload PCG (.wav) file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.success("✅ Audio uploaded successfully!")

    # Load audio with librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        y, sr = librosa.load(tmp_file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))

        st.subheader("🎧 Audio Features Extracted")
        st.write(f"**Duration**: {duration:.2f} seconds")
        st.write(f"**RMS Energy**: {rms:.6f}")

        # Button to analyze using ChatGPT
        if st.button("🧠 Analyze with ChatGPT"):
            with st.spinner("Analyzing with ChatGPT..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a skilled cardiologist analyzing heart sounds from phonocardiogram (PCG) audio data. "
                                    "Based on features like duration, energy, frequency content, and rhythm, provide a diagnostic impression."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"I have uploaded a PCG .wav file.\n"
                                    f"Extracted Features:\n"
                                    f"- Duration: {duration:.2f} seconds\n"
                                    f"- RMS Energy: {rms:.6f}\n\n"
                                    f"Based on these and common PCG interpretations (e.g., murmur, gallop, stenosis, regurgitation), give a detailed diagnostic analysis."
                                ),
                            },
                        ],
                        temperature=0.3,
                        max_tokens=600,
                    )

                    result = response.choices[0].message.content
                    st.success("✅ Diagnostic Impression Received!")
                    st.subheader("🩺 ChatGPT Analysis")
                    st.write(result)

                except Exception as e:
                    st.error(f"❌ Error during ChatGPT analysis: {e}")

    except Exception as err:
        st.error(f"❌ Could not process audio: {err}")
