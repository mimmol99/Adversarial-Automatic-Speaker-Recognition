import time
import wave

def get_wav_duration(wav_path):
    with wave.open(wav_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration
