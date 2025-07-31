import streamlit as st
from PIL import Image
import numpy as np
import openai
import os

# Set page config
st.set_page_config(page_title="PCG Analyzer with ChatGPT", layout="centered")

# Set OpenAI API Key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Title
st.title("🫀 PCG Waveform Analyzer")
st.markdown("Upload a **Phonocardiogram waveform image** (JPG/PNG), and get AI-assisted murmur diagnosis using ChatGPT.")

# File uploader
uploaded_file = st.file_uploader("Upload waveform image", type=["png", "jpg", "jpeg"])

def extract_waveform_features(image):
    # Convert image to grayscale using PIL
    image = image.convert("L")  # L mode = grayscale
    img_array = np.array(image)

    # Simple thresholding (invert)
    binary = np.where(img_array < 128, 1, 0)

    # Sum over vertical axis to detect waveform density
    vertical_sum = np.sum(binary, axis=0)
    peak_count = np.count_nonzero(vertical_sum > np.mean(vertical_sum))

    # Heuristic analysis based on peak density
    if peak_count < 10:
        return "Normal heart sound pattern detected, likely no murmur."
    elif 10 <= peak_count <= 20:
        return "Regular systolic murmur pattern, possible aortic stenosis."
    elif 20 < peak_count <= 30:
        return "Longer murmur pattern, possibly mitral regurgitation or VSD."
    else:
        return "Complex waveform with frequent murmur patterns; likely pathological."

def get_chatgpt_diagnosis(observation_text):
    prompt = f"""Analyze this PCG waveform description and give a medical interpretation.
    
Waveform Summary:
\"{observation_text}\"

List possible diagnoses (like aortic stenosis, mitral regurgitation, normal, etc.) and explain why.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["choices"][0]["message"]["content"]

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded PCG waveform", use_column_width=True)
    image = Image.open(uploaded_file)

    with st.spinner("🔍 Analyzing waveform..."):
        observation = extract_waveform_features(image)
        st.success("✅ Waveform Analysis Complete")
        st.markdown(f"**Pattern Observation:** {observation}")

    with st.spinner("🤖 Getting diagnosis from ChatGPT..."):
        diagnosis = get_chatgpt_diagnosis(observation)
        st.subheader("🧠 ChatGPT Diagnosis Suggestion")
        st.markdown(diagnosis)
