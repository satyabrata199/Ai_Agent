import requests

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"  # required by schema

def run_llm(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2
    }

    response = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]