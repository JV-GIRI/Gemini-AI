import streamlit as st
import librosa
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DeepSeek model from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# Extract basic PCG features
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

# Generate diagnosis using DeepSeek
def generate_diagnosis(observations):
    prompt = f"""You are a medical AI trained to diagnose heart sounds based on phonocardiogram features.
The following features were extracted from the patient's heart sound: {observations.tolist()}.
What is the most likely diagnosis? Provide reasoning."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("AI Diagnosis of PCG using DeepSeek")
uploaded_file = st.file_uploader("Upload PCG (.wav) file", type=["wav"])

if uploaded_file:
    with st.spinner("Analyzing..."):
        features = extract_features(uploaded_file)
        diagnosis = generate_diagnosis(features)
        st.success("Diagnosis Complete:")
        st.write(diagnosis)
