

# 🧱 PROJECT: `voice_agent_local`

## 🎯 What we’re building (recap)

A **local conversational AI** using:

* LM Studio (LLM)
* STT (speech → text)
* TTS (text → speech)

Clean structure:

* virtual environment
* requirements.txt
* modular code

---

# 🔹 STEP 0 — CREATE PROJECT FOLDER

Choose where you want it. Example:

```powershell
cd Desktop
mkdir voice_agent_local
cd voice_agent_local
```

Everything happens **inside this folder**.

---

# 🔹 STEP 1 — CREATE VIRTUAL ENVIRONMENT

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

You should see:

```text
(venv) PS C:\...\voice_agent_local>
```

✔️ Never skip this.

---

# 🔹 STEP 2 — PROJECT STRUCTURE (IMPORTANT)

Create this structure **exactly**:

```text
voice_agent_local/
│
├─ venv/
├─ requirements.txt
├─ README.md
│
└─ app/
   ├─ main.py
   ├─ conversation.py
   │
   ├─ llm/
   │  └─ lmstudio.py
   │
   ├─ stt/
   │  └─ whisper_stt.py
   │
   ├─ tts/
   │  └─ piper_tts.py
   │
   └─ utils/
      └─ audio.py
```

Create folders first, then empty `.py` files.

---

# 🔹 STEP 3 — `requirements.txt`

Create `requirements.txt` (top-level):

```txt
requests
sounddevice
numpy
faster-whisper
```

Install everything:

```powershell
pip install -r requirements.txt
```

If this fails → stop and fix before continuing.

---

# 🔹 STEP 4 — LLM ENGINE (LM STUDIO)

### `app/llm/lmstudio.py`

```python
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
```

👉 Make sure **LM Studio API server is running**.

---

# 🔹 STEP 5 — SPEECH TO TEXT (STT)

### `app/stt/whisper_stt.py`

```python
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
```

We keep it **blocking + simple** for now.

---

# 🔹 STEP 6 — TEXT TO SPEECH (TTS)

### Folder setup

```text
voice_agent_local/
└─ piper/
   ├─ piper.exe
   └─ voices/
      └─ en_US-amy-low.onnx
```

(You already know how to get Piper.)

---

### `app/tts/piper_tts.py`

```python
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
```

Playback comes next (we’ll add it).

---

# 🔹 STEP 7 — CONVERSATION LOOP (TEXT → VOICE)

### `app/conversation.py`

```python
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
```

---

# 🔹 STEP 8 — ENTRY POINT

### `app/main.py`

```python
from conversation import voice_chat

if __name__ == "__main__":
    voice_chat()
```

Run it:

```powershell
python app/main.py
```

---

# ✅ IF THIS WORKS, YOU NOW HAVE

* clean project layout
* isolated venv
* reproducible installs
* modular agent
* voice in / voice out
* LLM swappable anytime

This is **real engineering**, not demos.

---

# 🔜 NEXT PHASE OPTIONS

Say one and we continue **inside this structure**:

1️⃣ Add **interrupt / barge-in**
2️⃣ Stream audio instead of 5-second chunks
3️⃣ Plug this into your **agent loop + tools**
4️⃣ Add **memory (files / summaries)**
5️⃣ Move to **LiveKit architecture**

You’re doing this the *right* way.


Good call. This step is **critical** — if STT/TTS aren’t verified in isolation, everything above them becomes painful to debug.

We’ll do this **cleanly, step-by-step**:

1. Download
2. Verify files
3. Run **standalone tests** (no agent, no LLM)

No shortcuts.

---

# 🔊 PART A — STT (Speech → Text)

### Using **faster-whisper** (local, free, efficient)

---

## ✅ A1. Install dependencies (inside venv)

Make sure your venv is active:

```powershell
venv\Scripts\activate
```

Install STT deps:

```powershell
pip install faster-whisper sounddevice numpy
```

Verify install:

```powershell
pip show faster-whisper
```

If it shows version info → OK.

---

## ✅ A2. Download Whisper model (automatic)

You **do not manually download** Whisper models.

The first time you run:

```python
WhisperModel("small")
```

It auto-downloads to:

```text
C:\Users\<you>\.cache\huggingface\hub\
```

This is correct behavior.

---

## ✅ A3. STT Standalone Test (IMPORTANT)

Create **temporary test file**:

### `app/stt/test_stt.py`

```python
from whisper_stt import speech_to_text

print("Say something...")
text = speech_to_text()
print("Recognized:", text)
```

Run:

```powershell
python app/stt/test_stt.py
```

### ✅ SUCCESS looks like:

* mic records ~5 seconds
* text prints correctly

### ❌ If it fails:

* No mic → check Windows mic permissions
* CUDA error → change `device="cpu"` in `whisper_stt.py`

---

# 🔉 PART B — TTS (Text → Speech)

### Using **Piper** (offline, fast, free)

---

## ✅ B1. Download Piper (Windows)

Official releases:
👉 [https://github.com/rhasspy/piper/releases](https://github.com/rhasspy/piper/releases)

Download:

```text
piper_windows_amd64.zip
```

Extract to:

```text
voice_agent_local/piper/
```

You must have:

```text
piper/
├─ piper.exe
```

---

## ✅ B2. Download Voice Model

Official voices:
👉 [https://huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

Example (recommended):

```text
en_US-amy-low.onnx
```

Place here:

```text
piper/voices/en_US-amy-low.onnx
```

Your structure must be:

```text
voice_agent_local/
├─ piper/
│  ├─ piper.exe
│  └─ voices/
│     └─ en_US-amy-low.onnx
```

---

## ✅ B3. Verify Piper Manually (NO PYTHON YET)

From project root:

```powershell
echo Hello from Piper | piper\piper.exe -m piper\voices\en_US-amy-low.onnx -f test.wav
```

Then play `test.wav`.

✔️ If you hear voice → Piper is correct
❌ If not → stop here and fix paths

---

## ✅ B4. Python TTS Standalone Test

Create:

### `app/tts/test_tts.py`

```python
from piper_tts import speak

speak("Piper text to speech is working.")
print("Audio generated.")
```

Run:

```powershell
python app/tts/test_tts.py
```

Then play `output.wav`.

---

# 🔎 FINAL VERIFICATION CHECKLIST

Before moving on, confirm:

| Component             | Status |
| --------------------- | ------ |
| Microphone works      | ✅      |
| STT prints text       | ✅      |
| Piper CLI speaks      | ✅      |
| Piper Python works    | ✅      |
| No agent involved yet | ✅      |

If **any** box is ❌ → we stop and fix that part only.

---

# 🧠 WHY WE DID THIS

You just learned a **core systems lesson**:

> Always validate I/O subsystems independently before orchestration.

Most people skip this and suffer later.

---

## 🔜 NEXT (YOU CHOOSE)


Great question — and this is one of those places where **“better” depends on *what you’re optimizing for***.

I’ll give you a **clear, opinionated answer**, grounded in:

* your hardware (GTX 1660, ~6 GB VRAM, 8 GB RAM)
* offline / LM Studio
* conversational agent (voice)
* learning-by-doing (not just chatting)

No marketing fluff.

---

# 🧠 FIRST: WHAT “BETTER” MEANS FOR *YOU*

For your project, the LLM must be:

1. **Fast** (voice = latency matters)
2. **Stable** (no rambling, no infinite loops)
3. **Instruction-following** (agent-ready)
4. **Fits in VRAM** (or hybrid GPU/CPU without pain)
5. **Free & local**

So we care less about “benchmarks” and more about:

> *Does it behave well inside an agent loop and conversation?*

---

# 🏆 RECOMMENDED MODELS (LM Studio)

## 🥇 **Qwen 2.5 3B Instruct (GGUF / Q4–Q5)**

**Best overall choice for you**

### Why this one wins:

* Excellent instruction following
* Very stable in agent loops
* Low hallucination
* Strong conversational tone
* Designed for tool / agent use
* Works great at **3B** (huge deal)

### What to load in LM Studio:

* `Qwen2.5-3B-Instruct`
* Quantization: **Q4_K_M** or **Q5_K_M**

✅ Fits easily on GTX 1660
✅ Fast enough for voice
✅ Scales later to tools & planning

👉 **This should be your default model.**

---

## 🥈 **Llama 3.2 3B Instruct**

**Very good, slightly less disciplined**

### Pros:

* Natural conversation
* Good reasoning for size
* Stable responses

### Cons:

* Slightly weaker at strict instruction schemas
* More “chatty” unless you prompt carefully

Good for:

* friendly voice assistants
* casual conversation

Less ideal for:

* tight agent control loops (but still usable)

---

## 🥉 **Phi-3 Mini (3.8B)**

**Best reasoning-per-token, but…**

### Pros:

* Very strong logic
* Clean responses
* Efficient

### Cons:

* Dry personality
* Less conversational warmth
* Can feel robotic in voice mode

Good for:

* task agents
* logic-heavy tools

Less good for:

* friendly voice AI

---

# 🚫 MODELS I DO *NOT* RECOMMEND (FOR NOW)

❌ **7B models**
→ Too slow, VRAM pressure, voice latency suffers

❌ **Mistral 7B (any flavor)**
→ Great text model, poor agent discipline locally

❌ **Gemma 2B**
→ Weak instruction following

❌ **Unquantized / FP16**
→ Waste of your hardware

---

# ⚙️ LM STUDIO SETTINGS (IMPORTANT)

For **Qwen 2.5 3B**, use:

* **Context length**: 4096
* **Temperature**: `0.2 – 0.4`
* **Top-p**: `0.9`
* **GPU layers**: as high as possible (auto is fine)
* **Stop sequences** (optional):

  ```
  </s>
  ```

This makes it:

* deterministic
* calm
* agent-safe

---

# 🧠 WHY THIS MODEL WORKS WELL WITH YOUR AGENT LOOP

Qwen models are trained with:

* explicit instruction schemas
* tool / agent-style reasoning
* low verbosity bias

That means:

* it stops when told
* it doesn’t fight your loop
* it behaves predictably

This matters **more than raw intelligence**.

---

# ✅ FINAL RECOMMENDATION (TL;DR)

If you want **one model to commit to**:

> **Qwen2.5-3B-Instruct (Q4_K_M)**
> via **LM Studio**

Use it for:

* conversation
* agent loops
* tools
* memory
* later LiveKit integration

---

## 🔜 NEXT STEP (LOGICAL ORDER)

Next, we should:

1. **Lock the model**
2. Tune **system prompt for voice**
3. Add **interrupt / barge-in**
4. Then plug it back into the **agent loop**

Great question — and this is one of those places where **“better” depends on *what you’re optimizing for***.

I’ll give you a **clear, opinionated answer**, grounded in:

* your hardware (GTX 1660, ~6 GB VRAM, 8 GB RAM)
* offline / LM Studio
* conversational agent (voice)
* learning-by-doing (not just chatting)

No marketing fluff.

---

# 🧠 FIRST: WHAT “BETTER” MEANS FOR *YOU*

For your project, the LLM must be:

1. **Fast** (voice = latency matters)
2. **Stable** (no rambling, no infinite loops)
3. **Instruction-following** (agent-ready)
4. **Fits in VRAM** (or hybrid GPU/CPU without pain)
5. **Free & local**

So we care less about “benchmarks” and more about:

> *Does it behave well inside an agent loop and conversation?*

---

# 🏆 RECOMMENDED MODELS (LM Studio)

## 🥇 **Qwen 2.5 3B Instruct (GGUF / Q4–Q5)**

**Best overall choice for you**

### Why this one wins:

* Excellent instruction following
* Very stable in agent loops
* Low hallucination
* Strong conversational tone
* Designed for tool / agent use
* Works great at **3B** (huge deal)

### What to load in LM Studio:

* `Qwen2.5-3B-Instruct`
* Quantization: **Q4_K_M** or **Q5_K_M**

✅ Fits easily on GTX 1660
✅ Fast enough for voice
✅ Scales later to tools & planning

👉 **This should be your default model.**

---

## 🥈 **Llama 3.2 3B Instruct**

**Very good, slightly less disciplined**

### Pros:

* Natural conversation
* Good reasoning for size
* Stable responses

### Cons:

* Slightly weaker at strict instruction schemas
* More “chatty” unless you prompt carefully

Good for:

* friendly voice assistants
* casual conversation

Less ideal for:

* tight agent control loops (but still usable)

---

## 🥉 **Phi-3 Mini (3.8B)**

**Best reasoning-per-token, but…**

### Pros:

* Very strong logic
* Clean responses
* Efficient

### Cons:

* Dry personality
* Less conversational warmth
* Can feel robotic in voice mode

Good for:

* task agents
* logic-heavy tools

Less good for:

* friendly voice AI

---

# 🚫 MODELS I DO *NOT* RECOMMEND (FOR NOW)

❌ **7B models**
→ Too slow, VRAM pressure, voice latency suffers

❌ **Mistral 7B (any flavor)**
→ Great text model, poor agent discipline locally

❌ **Gemma 2B**
→ Weak instruction following

❌ **Unquantized / FP16**
→ Waste of your hardware

---

# ⚙️ LM STUDIO SETTINGS (IMPORTANT)

For **Qwen 2.5 3B**, use:

* **Context length**: 4096
* **Temperature**: `0.2 – 0.4`
* **Top-p**: `0.9`
* **GPU layers**: as high as possible (auto is fine)
* **Stop sequences** (optional):

  ```
  </s>
  ```

This makes it:

* deterministic
* calm
* agent-safe

---

# 🧠 WHY THIS MODEL WORKS WELL WITH YOUR AGENT LOOP

Qwen models are trained with:

* explicit instruction schemas
* tool / agent-style reasoning
* low verbosity bias

That means:

* it stops when told
* it doesn’t fight your loop
* it behaves predictably

This matters **more than raw intelligence**.

---

# ✅ FINAL RECOMMENDATION (TL;DR)

If you want **one model to commit to**:

> **Qwen2.5-3B-Instruct (Q4_K_M)**
> via **LM Studio**

Use it for:

* conversation
* agent loops
* tools
* memory
* later LiveKit integration

---

## 🔜 NEXT STEP (LOGICAL ORDER)

Next, we should:

1. **Lock the model**
2. Tune **system prompt for voice**
3. Add **interrupt / barge-in**
4. Then plug it back into the **agent loop**

Say:

* **“Lock Qwen and tune prompt”**
  or
* **“Compare two models live”**

You’re building this the *right* way — slow, solid, reusable.

