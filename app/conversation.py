from llm.lmstudio import run_llm
from stt.whisper_stt import speech_to_text
from tts.piper_tts import speak

def voice_chat():
    messages = [
        {
            "role": "system",
            "content": "You are a friendly, concise conversational assistant."
        }
    ]

    while True:
        print("🎧 Listening...")
        user_text = speech_to_text()

        if not user_text:
            continue

        print("You:", user_text)
        messages.append({"role": "user", "content": user_text})

        reply = run_llm(messages)
        print("AI:", reply)

        speak(reply)
        messages.append({"role": "assistant", "content": reply})