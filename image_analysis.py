import torch
import streamlit as st
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

FLORENCE_MODEL = "microsoft/Florence-2-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32


# --------------------------------------------------
# Load Florence-2
# --------------------------------------------------


@st.cache_resource
def load_florence():

    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)

    return processor, model


# --------------------------------------------------
# Image analysis
# --------------------------------------------------


def analyze_image(image):

    processor, model = load_florence()

    prompt = "The image shows a collection of electronic components including a DC motor, wires, LEDs, a battery, cardboard and adhesive tape arranged on a table."

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        device, torch_dtype
    )

    with torch.no_grad():

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    result = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )

    return result
