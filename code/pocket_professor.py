# Pocket Professor — Raspberry Pi Safe Edition
# Simplified for Python 3.11 on Raspberry Pi 5
# Uses: Vosk (speech), llama.cpp (LLM/VLM), espeak (TTS), picamera2 (optional)

import os
import io
import time
import json
import base64
import subprocess

import numpy as np
import sounddevice as sd
from PIL import Image

from vosk import Model, KaldiRecognizer
from llama_cpp import Llama
from picamera2 import Picamera2

# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

MAIN_MODEL_PATH = "models/main_model.gguf"
MMPROJ_PATH     = "models/mmproj.gguf"
VOSK_MODEL_PATH = "models/vosk/vosk-model-small-en-us-0.15"

SAMPLE_RATE           = 16000
VISION_RECORD_SECONDS = 2
WAKE_CHUNK_SECONDS    = 2       # seconds per wake-word poll chunk
FOLLOWUP_SECONDS      = 4       # how long to listen for a follow-up

SILENCE_THRESHOLD     = 350     # RMS below this = silence
SILENCE_TIMEOUT       = 2.2     # seconds of silence before stopping
MAX_RECORD_SECONDS    = 15      # hard cap on any single recording
CHUNK_SECONDS         = 0.5     # size of each audio chunk

CAMERA_WIDTH          = 640
CAMERA_HEIGHT         = 480
MAX_HINT_TOKENS       = 80
PHOTO_DELAY_SECONDS   = 2.0

# Words/phrases that trigger wake-up
WAKE_WORDS = {"professor", "pocket", "hey professor", "hey pocket"}

# Words that indicate the user is done
DONE_WORDS = {"no", "nope", "nah", "nothing", "done", "thanks", "thank you", "goodbye", "bye"}

# ─────────────────────────────────────────────────────────────
# Text‑to‑Speech using espeak
# ─────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    print(f"\nProfessor: {text}")
    subprocess.run(["espeak", text])

# ─────────────────────────────────────────────────────────────
# Speech‑to‑Text using Vosk
# ─────────────────────────────────────────────────────────────

print("Loading Vosk speech model...")
vosk_model = Model(VOSK_MODEL_PATH)
print("Vosk ready.")


def listen(max_seconds: int = MAX_RECORD_SECONDS) -> str:
    """Record until silence is detected or the hard cap is reached."""
    print("Listening...")

    chunks = []
    silent_for = 0.0
    chunk_size = int(CHUNK_SECONDS * SAMPLE_RATE)

    while True:
        chunk = sd.rec(chunk_size, samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        chunks.append(chunk)

        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

        if rms < SILENCE_THRESHOLD:
            silent_for += CHUNK_SECONDS
        else:
            silent_for = 0.0

        total_recorded = len(chunks) * CHUNK_SECONDS

        if silent_for >= SILENCE_TIMEOUT and total_recorded > CHUNK_SECONDS:
            break

        if total_recorded >= max_seconds:
            break

    audio = np.concatenate(chunks, axis=0)

    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    rec.AcceptWaveform(audio.tobytes())
    result = json.loads(rec.Result())

    text = result.get("text", "").strip()
    print(f"Heard: {text}")
    return text


def wait_for_wake_word() -> None:
    """Block until a wake word is detected, then return."""
    print("Waiting for wake word...")
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    while True:
        audio = sd.rec(int(WAKE_CHUNK_SECONDS * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='int16')
        sd.wait()

        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.Result())
        heard = result.get("text", "").lower()

        if any(wake in heard for wake in WAKE_WORDS):
            print(f"Wake word detected in: '{heard}'")
            return


def is_done(text: str) -> bool:
    """Return True if the user signals they have no more questions."""
    if not text:
        return True
    words = set(text.lower().split())
    return bool(words & DONE_WORDS)

# ─────────────────────────────────────────────────────────────
# Yes/No parsing
# ─────────────────────────────────────────────────────────────

YES_WORDS = {"yes", "yeah", "yep", "sure", "okay", "ok"}
NO_WORDS  = {"no", "nope", "nah", "skip"}


def parse_yes_no(text: str) -> bool | None:
    words = set(text.lower().split())
    if words & YES_WORDS:
        return True
    if words & NO_WORDS:
        return False
    return None

# ─────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────

def init_camera():
    try:
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start(show_preview=False)
        time.sleep(1)
        print("Camera ready.")
        return cam
    except Exception as e:
        print("Camera unavailable:", e)
        return None


def capture_image_b64(camera):
    try:
        frame = camera.capture_array()
        img = Image.fromarray(frame, mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print("Camera capture failed:", e)
        return None

# ─────────────────────────────────────────────────────────────
# LLM / VLM
# ─────────────────────────────────────────────────────────────

print("Loading language/vision model...")
llm = Llama(
    model_path=MAIN_MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
)
print("Model ready.")

SYSTEM_PROMPT = (
    "You are Pocket Professor, a practical IT tutor helping students troubleshoot real problems. "
    "Give one specific, actionable hint — not a metaphor or vague observation. "
    "Examples of good hints: 'Check if the power cable is firmly plugged into the wall.' "
    "'Try holding the power button for 10 seconds.' "
    "'Check if the battery LED lights up when plugged in.' "
    "Never say things like 'you might not be able to see the problem' — always suggest a concrete action to take."
)


def clean_chatml(text: str) -> str:
    for token in ["<|im_start|>", "<|im_end|>", "</code>", "<code>"]:
        text = text.replace(token, "")
    return text.strip()


def get_hint(question: str, image_b64: str | None) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Student question: {question}\n\n"
        f"Hint:"
    )

    response = llm(
        prompt,
        max_tokens=MAX_HINT_TOKENS,
        temperature=0.7,
        repeat_penalty=1.2,
        stop=["Student:", "\n\n"],
    )

    raw = response["choices"][0].get("text", "").strip()
    print(f"Raw model output: {repr(raw)}")

    if not raw:
        return "I could not come up with a hint. Try rephrasing your question."

    hint = clean_chatml(raw)
    return hint if hint else "I got an empty response. Please try again."

# ─────────────────────────────────────────────────────────────
# Question handler
# ─────────────────────────────────────────────────────────────

def handle_question(question: str, camera) -> None:
    """Optionally capture an image, then speak a hint."""
    image_b64 = None

    if camera:
        speak("Should I take a look?")
        decision = parse_yes_no(listen(VISION_RECORD_SECONDS))

        if decision:
            speak("Hold it up.")
            time.sleep(PHOTO_DELAY_SECONDS)
            image_b64 = capture_image_b64(camera)

    speak("Let me think.")
    hint = get_hint(question, image_b64)
    speak(hint)

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def main():
    camera = init_camera()

    speak("Pocket Professor is ready.")

    while True:
        try:
            # ── Idle: wait for wake word ──────────────────────
            wait_for_wake_word()

            # ── Activated: handle first question ─────────────
            speak("Go ahead.")
            question = listen()

            if not question:
                speak("I did not hear anything.")
                continue

            handle_question(question, camera)

            # ── Follow-up loop ────────────────────────────────
            while True:
                speak("Anything else?")
                followup = listen(FOLLOWUP_SECONDS)

                if is_done(followup):
                    speak("Good luck. Say hey Pocket Professor whenever you need me.")
                    break

                handle_question(followup, camera)

        except KeyboardInterrupt:
            print("Exiting...")
            if camera:
                camera.stop()
            break

        except Exception as e:
            print("Error:", e)
            speak("Something went wrong.")


if __name__ == "__main__":
    main()
