import io
import librosa
import streamlit as st
import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

WHISPER_MODEL = "openai/whisper-tiny"


# --------------------------------------------------
# Load Whisper
# --------------------------------------------------


@st.cache_resource
def load_whisper():

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL, torch_dtype=torch_dtype
    ).to(device)

    return processor, model


# --------------------------------------------------
# Speech recognition
# --------------------------------------------------


def transcribe_audio(audio_file):

    processor, model = load_whisper()

    audio_bytes = audio_file.read()

    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)

    with torch.no_grad():

        generated_ids = model.generate(input_features, max_new_tokens=100)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription
