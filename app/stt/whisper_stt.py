import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cuda",        # change to "cpu" if needed
    compute_type="float16"
)

SAMPLE_RATE = 16000

def record_audio(seconds=5):
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    return audio.flatten()

def speech_to_text():
    audio = record_audio()
    segments, _ = model.transcribe(audio)
    return " ".join(seg.text for seg in segments).strip()