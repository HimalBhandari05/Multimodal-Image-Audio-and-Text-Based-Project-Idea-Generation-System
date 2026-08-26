# 💡 Multimodal Project Idea Generator

A multimodal AI application that generates realistic engineering and DIY project ideas based on the physical materials you have and your custom requirements. It processes **images** of available components, **voice** or **text** descriptions, and uses a local LLM to propose feasible projects.

---

## 🚀 Key Features & Pipeline

| Stage | Technology / Model | Purpose |
| :--- | :--- | :--- |
| **Vision Analysis** | `microsoft/Florence-2-base` | Detects and identifies components from uploaded photos |
| **Speech-to-Text** | `openai/whisper-tiny` | Transcribes recorded voice prompts into text |
| **Project Generation** | `qwen2.5:3b` (via Ollama) | Proposes structured, realistic project ideas with steps |
| **User Interface** | `Streamlit` | Interactive web GUI for image upload, audio input, and results |

---

## 📋 Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed and running
- *(Optional)* NVIDIA GPU with CUDA support for faster inference (CPU is automatically supported)

---

## ⚙️ Installation & Setup

### 1. Clone the repository & create a virtual environment

```bash
git clone https://github.com/HimalBhandari05/Multimodal-Image-Audio-and-Text-Based-Project-Idea-Generation-System.git
cd Multimodal-Image-Audio-and-Text-Based-Project-Idea-Generation-System

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install streamlit torch transformers librosa pillow requests torchvision einops timm soundfile
```

### 3. Set up the local LLM with Ollama

Make sure Ollama is running and pull the Qwen model:

```bash
ollama serve
ollama pull qwen2.5:3b
```

---

## 🎯 How to Run

1. **Launch the Streamlit app:**
   ```bash
   streamlit run GUI.py
   ```

2. **Using the app:**
   1. **Upload an image** containing your available hardware/materials (e.g., motors, LEDs, sensors, cardboard).
   2. **Provide your requirement** by speaking into the microphone or typing a prompt (e.g., *"Suggest a beginner project taking under 1 hour"*).
   3. Click **"Generate Project Ideas"** to get tailored project proposals.

---

## 📂 Project Structure

```text
├── GUI.py                 # Streamlit front-end application
├── image_analysis.py      # Image processing using Florence-2
├── speech_recognition.py  # Audio transcription using Whisper-tiny
├── generate_project.py    # Prompt builder and Ollama API integration
├── notebooks/             # Exploration and testing notebooks
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify it!
