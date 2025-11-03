import sounddevice as sd
from scipy.io.wavfile import write
import openai
import numpy as np
import tempfile
import threading
from settings import settings

# SET YOUR OPENAI API KEY
openai.api_key = settings["openAIToken"]
SAMPLE_RATE = 16000

# Globals for recording state
_audio_buffer = []
_stream = None

def _audio_callback(indata, frames, time, status):
    _audio_buffer.append(indata.copy())

def start_recording():
    global _audio_buffer, _stream
    _audio_buffer = []
    _stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_audio_callback)
    _stream.start()
    print("Recording started... Press stop to finish.")

def stop_recording(filename=None):
    global _audio_buffer, _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None
    if not _audio_buffer:
        raise RuntimeError("No audio was recorded!")
    audio_data = np.concatenate(_audio_buffer, axis=0)
    if filename is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        filename = tmp.name
    write(filename, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    print(f"Recording stopped and saved to {filename}")
    return filename

def transcribe_audio(filename):
    print("Transcribing audio...")
    with open(filename, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
    print("Transcription:")
    print(transcript.text)
    return transcript.text

# CLI/Test example: (Replace with your GUI calls)
if __name__ == "__main__":
    import time
    start_recording()
    time.sleep(5)  # Replace this with your GUI's stop trigger
    try:
        wav_file = stop_recording()
        transcribe_audio(wav_file)
    except RuntimeError as e:
        print("Error:", e)
