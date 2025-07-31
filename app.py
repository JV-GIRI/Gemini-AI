import streamlit as st
import openai
import os

# Set your OpenAI API key (DO NOT expose in production!)
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=openai_api_key)

# Streamlit UI
st.set_page_config(page_title="PCG Analyzer with ChatGPT", layout="wide")
st.title("🫀 PCG Waveform Analyzer with ChatGPT")
st.write("Upload a PCG waveform file (.txt or .csv) to get a diagnostic interpretation.")

uploaded_file = st.file_uploader("Upload PCG waveform file", type=["txt", "csv"])

if uploaded_file:
    waveform_data = uploaded_file.read().decode("utf-8")

    st.subheader("📊 Uploaded Waveform Data Preview")
    st.code(waveform_data[:500])  # Show first 500 characters

    if st.button("🧠 Analyze with ChatGPT"):
        with st.spinner("Contacting ChatGPT for diagnosis..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a cardiologist expert in interpreting Phonocardiogram (PCG) signals."},
                        {"role": "user", "content": f"Analyze the following PCG waveform and provide a diagnostic impression:\n\n{waveform_data}"}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                diagnosis = response.choices[0].message.content
                st.success("✅ Diagnosis Received!")
                st.subheader("🩺 Diagnostic Impression")
                st.write(diagnosis)

            except Exception as e:
                st.error(f"❌ Error: {e}")
