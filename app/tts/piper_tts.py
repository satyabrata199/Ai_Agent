import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PIPER_EXE = BASE_DIR / "piper" / "piper.exe"
VOICE = BASE_DIR / "piper" / "voices" / "en_US-amy-low.onnx"
OUTPUT = BASE_DIR / "output.wav"

def speak(text: str):
    subprocess.run(
        [
            str(PIPER_EXE),
            "-m", str(VOICE),
            "-f", str(OUTPUT)
        ],
        input=text.encode("utf-8"),
        check=True
    )