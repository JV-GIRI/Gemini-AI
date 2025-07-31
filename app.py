import streamlit as st
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import io

# --- App Configuration ---
st.set_page_config(
    page_title="AI PCG Diagnosis (Research Concept)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CRITICAL DISCLAIMER ---
st.title("🫀 AI Phonocardiography Analysis")
st.warning(
    "**RESEARCH PROOF-OF-CONCEPT ONLY.** This app uses a **simulated AI logic engine**. "
    "The diagnosis is NOT REAL and is chosen from pre-written templates for demonstration purposes. "
    "It must be validated against echocardiography.",
    icon="⚠️"
)


# --- SIMULATED AI DIAGNOSTIC ENGINE ---
def get_simulated_diagnosis(audio_data, sample_rate, valve_name):
    std_dev = np.std(audio_data)
    peak_amplitude = np.max(np.abs(audio_data))

    report_normal = (
        "**Likely Diagnosis:** Normal Heart Sounds\n\n"
        "**Analysis:** S1 and S2 sounds are distinct and clear. The systolic and diastolic "
        "periods are acoustically silent. No significant murmurs, gallops, or rubs detected. "
        "The waveform is consistent with normal cardiac function."
    )
    report_as = (
        "**Likely Diagnosis:** Aortic Stenosis (AS)\n\n"
        "**Analysis:** The waveform shows a prominent **midsystolic murmur** with a "
        "**crescendo-decrescendo ('diamond-shaped') pattern**. This is the pathognomonic "
        "sign of turbulent blood flow across a narrowed aortic valve during systole. "
        "S2 may be diminished. This finding is highly relevant for RVHD studies."
    )
    report_ms = (
        "**Likely Diagnosis:** Mitral Stenosis (MS)\n\n"
        "**Analysis:** A **low-frequency, rumbling, mid-diastolic murmur** is present. "
        "This is the classic hallmark of turbulent blood flow across a narrowed mitral "
        "valve during ventricular filling. S1 is often accentuated. This is the most "
        "common valvular lesion in Rheumatic Heart Disease."
    )
    report_mr = (
        "**Likely Diagnosis:** Mitral Regurgitation (MR)\n\n"
        "**Analysis:** A **holosystolic (pansystolic), high-pitched, 'blowing' murmur** "
        "is detected, starting at S1 and continuing to S2. This is caused by blood "
        "leaking backward through an incompetent mitral valve during systole. The murmur "
        "may obscure the S2 sound."
    )

    st.info(f"Simulated Logic Trigger for {valve_name}: std_dev={std_dev:.0f}, peak={peak_amplitude:.0f}")

    if valve_name == "Aortic Valve":
        if std_dev > 3500:
            return report_as
        else:
            return report_normal
    elif valve_name == "Mitral Valve":
        if 1500 < std_dev <= 3500:
            return report_ms
        elif std_dev <= 1500 and peak_amplitude > 10000:
            return report_mr
        else:
            return report_normal
    else:
        return report_normal


# --- WAVEFORM PLOTTING ---
def plot_waveform(sample_rate, audio_data, valve_name, max_duration, amp_scale, noise_thresh):
    fig, ax = plt.subplots(figsize=(10, 2))
    duration = len(audio_data) / sample_rate
    time = np.linspace(0., duration, len(audio_data))

    # Apply amplitude scaling
    audio_data = audio_data * amp_scale

    # Apply basic noise reduction
    if noise_thresh > 0:
        audio_data = np.where(np.abs(audio_data) < noise_thresh, 0, audio_data)

    ax.plot(time, audio_data, lw=0.7)
    ax.set_title(f"{valve_name} Waveform")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, min(duration, max_duration))
    return fig


# --- SIDEBAR ---
st.sidebar.header("🧑‍⚕️ Patient Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=65.0)
phone = st.sidebar.text_input("Phone Number")

# BMI calculation
if height > 0:
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    st.sidebar.markdown(f"**BMI:** {bmi:.1f}")

# Controls for waveform visualization
st.sidebar.header("🎚️ Waveform Controls")
max_duration = st.sidebar.slider("Max Duration (s)", min_value=1, max_value=10, value=8)
amp_scale = st.sidebar.slider("Amplitude Scaling", min_value=0.1, max_value=3.0, value=1.0)
noise_thresh = st.sidebar.slider("Noise Reduction Threshold", min_value=0, max_value=1000, value=0)

st.sidebar.header("📖 About This Project")
st.sidebar.info(
    "This application is a conceptual tool for the research project: 'Prospective study "
    "of an application of AI in early detection of RVHD in asymptomatic individuals "
    "using a non-invasive method of Phonocardiography.'"
)
st.sidebar.header("⚠️ Disclaimer")
st.sidebar.error(
    "The AI diagnosis is simulated. The logic engine is a placeholder. "
    "All findings must be correlated with echocardiography."
)

# --- MAIN INTERFACE ---
st.header("1. Upload Patient PCG Files (.wav)")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

valve_files = {
    "Aortic Valve": col1.file_uploader("Aortic Valve (AV)", type=["wav"]),
    "Pulmonary Valve": col2.file_uploader("Pulmonary Valve (PV)", type=["wav"]),
    "Mitral Valve": col3.file_uploader("Mitral Valve (MV)", type=["wav"]),
    "Tricuspid Valve": col4.file_uploader("Tricuspid Valve (TV)", type=["wav"]),
}
st.markdown("---")

st.header("2. AI Analysis & Report")
if st.button("🔬 Generate Diagnostic Report", type="primary", use_container_width=True):
    if not any(valve_files.values()):
        st.error("Please upload at least one audio file to generate a report.")
    else:
        for valve_name, uploaded_file in valve_files.items():
            if uploaded_file is not None:
                st.subheader(f"Analysis for: {valve_name}")
                audio_buffer = io.BytesIO(uploaded_file.getvalue())
                try:
                    sample_rate, audio_data = wavfile.read(audio_buffer)

                    # Plot waveform with controls
                    fig = plot_waveform(sample_rate, audio_data, valve_name, max_duration, amp_scale, noise_thresh)
                    st.pyplot(fig)

                    # Display report
                    st.markdown("##### 🤖 AI-Generated Report")
                    diagnosis_report = get_simulated_diagnosis(audio_data, sample_rate, valve_name)
                    st.write(diagnosis_report)
                    st.markdown("---")

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}. Please ensure it is a valid WAV file.")
