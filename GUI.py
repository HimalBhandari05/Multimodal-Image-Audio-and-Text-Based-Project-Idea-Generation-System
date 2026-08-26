import requests
from PIL import Image
import streamlit as st

from image_analysis import analyze_image
from speech_recognition import transcribe_audio
from generate_project import generate_projects


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(page_title="Project Idea Generator", page_icon="💡", layout="wide")

st.title("Project Idea Generator")

st.write(
    "Upload a photo of your available components, "
    "describe what you want using your voice, "
    "and get project ideas generated from them."
)

st.divider()

# --------------------------------------------------
# Input section
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("1. Upload Materials")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with col2:

    st.subheader("2. Describe Your Requirement")

    audio_file = st.audio_input("Record your requirement")

    text_request = st.text_area(
        "Or type your requirement",
        placeholder=(
            "Example: Give me a beginner project " "that takes less than one hour."
        ),
    )

st.divider()

# --------------------------------------------------
# Generate
# --------------------------------------------------

if st.button("Generate Project Ideas", type="primary", use_container_width=True):

    if uploaded_image is None:

        st.error("Please upload an image first.")

        st.stop()

    # Load image

    image = Image.open(uploaded_image).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(image, use_container_width=True)

    # --------------------------------------------------
    # Florence
    # --------------------------------------------------

    with st.spinner("Analyzing the image..."):

        detection_result = analyze_image(image)

    st.subheader("Detected Materials")

    st.json(detection_result)

    # Convert Florence result into text

    materials = str(detection_result)

    # --------------------------------------------------
    # Whisper
    # --------------------------------------------------

    voice_request = ""

    if audio_file is not None:

        with st.spinner("Understanding your voice..."):

            voice_request = transcribe_audio(audio_file)

        st.subheader("Voice Requirement")

        st.write(voice_request)

    # --------------------------------------------------
    # Combine request
    # --------------------------------------------------

    if text_request.strip():

        user_request = text_request

    elif voice_request.strip():

        user_request = voice_request

    else:

        user_request = "Suggest beginner-friendly projects " "using these materials."

    # --------------------------------------------------
    # Ollama
    # --------------------------------------------------

    with st.spinner("Generating project ideas..."):

        try:

            projects = generate_projects(materials, user_request)

        except requests.exceptions.ConnectionError:

            st.error("Ollama is not running. " "Start it with: ollama serve")

            st.stop()

        except Exception as error:

            st.error(f"Error: {error}")

            st.stop()

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.subheader("Generated Project Ideas")

    st.markdown(projects)
