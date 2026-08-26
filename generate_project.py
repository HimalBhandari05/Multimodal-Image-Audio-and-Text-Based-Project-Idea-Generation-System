import requests

OLLAMA_MODEL = "qwen2.5:3b"

def generate_projects(materials, user_request):

    prompt = f"""
You are an engineering project idea generator.

The user has provided a photo of their available materials.

Detected materials:
{materials}

User request:
{user_request}

Suggest exactly 3 realistic projects.

Only use materials that are actually available.
Do not invent expensive or specialized components.

For each project provide:

Project Name:
Difficulty:
Estimated Time:
Materials Used:
How It Works:
Why It Is Suitable:

Keep the explanations short and practical.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )

    response.raise_for_status()

    return response.json()["response"]
