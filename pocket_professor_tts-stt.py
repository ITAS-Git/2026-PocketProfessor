# Pocket Professor — Main Program -
# Listens for spoken IT questions and responds with a single guiding hint.
# Run with: python pocket_professor.py
# Stop with: Ctrl+C

 
import os
import time
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import pyttsx3
from llama_cpp import Llama
 
#Settings
 
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
 
# Text to Speech 
 
def speak(text):
    """Speaks the given text out loud through the speaker."""
    print(f"Professor: {text}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
 
#Speech to Text 
 
def listen(stt_model):
    """
    Records 5 seconds of audio from the microphone and transcribes it.
    Returns the transcribed text, or None if nothing was heard.
    """
    print("\nListening... (speak your question now)")
 
    # Record audio from the microphone
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()  # Wait for recording to finish
    print("Processing your question...")
 
    #Save to a temporary file so Whisper can read it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, SAMPLE_RATE, audio)
        tmp_path = tmp.name
 
    try:
        result = stt_model.transcribe(tmp_path)
        question = result["text"].strip()
        if question:
            print(f"You asked: {question}")
            return question
        else:
            return None
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None
    finally:
        os.remove(tmp_path)  # Clean up temp file
 
# AI Hint Generation 
 
# Instructions that tell the AI how to behave
SYSTEM_PROMPT = """<|system|>
You are Pocket Professor, an IT support tutor. Your only job is to give ONE short guiding question that helps the student discover the answer themselves.
 
STRICT RULES:
- ONE sentence only. No lists. No explanations.
- Must be a question, not a statement.
- Point toward the most likely root cause.
- Be specific to the problem described.
 
GOOD examples:
Student: My two VMs can't ping each other.
Hint: Have you verified that both VMs are assigned unique IP addresses with no conflicts?
 
Student: My monitor won't turn on.
Hint: Have you checked that the display cable is firmly connected to both the monitor and the GPU?
 
Student: My Failover Cluster isn't recognizing my iSCSI disks.
Hint: Have you confirmed that the iSCSI initiator is connected and the target portal is properly configured on each node?
 
Student: My VMs keep bluescreening during live migration.
Hint: Have you checked whether the destination host has compatible CPU features and enough available memory for the migrating VM?
</s>"""
 
def get_hint(llm, question):
    """Sends the question to TinyLlama and returns a hint."""
    prompt = f"""{SYSTEM_PROMPT}
<|user|>
{question}
</s>
<|assistant|>
Hint:"""
 
    output = llm(
        prompt,
        max_tokens=60,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "<|user|>", "<|system|>", "\n\n"]
    )
    return output["choices"][0]["text"].strip()
 
# Main Program 
 
def main():
    # Load the AI model
    print("Loading AI model, please wait...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    print("AI model loaded!")
 
    # Load Whisper speech-to-text model
    print("Loading speech recognition model...")
    stt_model = whisper.load_model("tiny")
    print("Speech recognition ready!")
 
    # Welcome message
    speak("Hello! I am Pocket Professor. Ask me an IT question and I will give you a hint.")
 
    # Main loop, keep listening and responding until Ctrl+C
    while True:
        try:
            speak("Go ahead, ask your question.")
 
            # Listen for a question
            question = listen(stt_model)
 
            if not question:
                speak("I did not catch that. Please try again.")
                continue
 
            # Get a hint from the AI
            speak("Let me think about that.")
            start = time.time()
            hint = get_hint(llm, question)
            response_time = time.time() - start
 
            print(f"Response time: {response_time:.2f} seconds")
 
            # Speak the hint
            speak(hint)
 
        except KeyboardInterrupt:
            print("\nShutting down...")
            speak("Goodbye! Good luck with your studies.")
            break
        except Exception as e:
            print(f"Error: {e}")
            speak("Something went wrong. Let me try again.")
            time.sleep(1)
 
if __name__ == "__main__":
    main()
 