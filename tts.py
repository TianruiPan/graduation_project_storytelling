from settings import settings
from openai import OpenAI
from pathlib import Path
import tempfile
import os
from pydub import AudioSegment
from pydub.playback import play

client = OpenAI(api_key=settings["openAIToken"])

def say(text, voice='coral', model="gpt-4o-mini-tts"):
    # Generate speech and play it (blocking function, run in thread)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        speech_file_path = Path(tmp_file.name)

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    
    audio = AudioSegment.from_mp3(str(speech_file_path))
    play(audio)
    try:
        os.remove(speech_file_path)
    except Exception:
        pass

if __name__ == "__main__":
    say(
        "Today is a wonderful day to build something people love!",
        voice="coral"
    )