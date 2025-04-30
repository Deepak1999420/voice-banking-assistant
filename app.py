import os
from flask import send_from_directory, send_file
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import requests
import pyttsx3
import tempfile
import time

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize Whisper model (load once at startup)
modelASR = WhisperModel("small")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate

def record_audio():
    """Record audio until silence is detected"""
    fs = 16000  # Sample rate
    seconds_per_chunk = 2  # Duration of each recording chunk
    max_silence_duration = 1.0  # Stop after this much silence (seconds)
    recording = []

    print("Recording... Speak now!")
    time.sleep(1.5)
    print("speak really now!!")
    while True:
        chunk = sd.rec(int(seconds_per_chunk * fs), samplerate=fs, channels=1)
        sd.wait()
        recording.extend(chunk)
        print("#")
        
        # Check for silence to end recording
        if len(recording) > fs * max_silence_duration:
            recording_np = np.array(recording)
            print(".")
            if (np.max(recording_np[int(-fs * max_silence_duration):]) < 0.04) and (np.min(recording_np[int(-fs * max_silence_duration):]) > -0.04):
                break
    print("Recording stopped.")
    return fs, np.array(recording)

def transcribe_audio(audio_path):
    """Transcribe audio to text using Whisper"""
    segments, info = modelASR.transcribe(audio_path)
    return " ".join(segment.text for segment in segments)

def get_ai_response(prompt, api_key):
    """Get response from DeepSeek via OpenRouter"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://voice-assistant-for-blind.com",
        "X-Title": "Blind Assistant"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

def text_to_speech(text, output_path):
    """Convert text to speech and save as MP3"""
    engine.save_to_file(text, output_path)
    engine.runAndWait()

@app.route('/api/process', methods=['POST'])
def process_audio():
    try:
        # 1. Record audio
        fs, audio_data = record_audio()
        
        # 2. Save temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_wav:
            write(tmp_wav.name, fs, audio_data)
            
            # 3. Transcribe audio to text
            user_input = transcribe_audio(tmp_wav.name)
            print(f"User said: {user_input}")
            
            # 4. Get AI response
            OPENROUTER_KEY = "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"  # Replace with your key
            ai_response = get_ai_response(user_input, OPENROUTER_KEY)
            print(f"AI Response: {ai_response}")
            
            # 5. Convert response to speech
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as tmp_mp3:
                text_to_speech(ai_response, tmp_mp3.name)
                
                # Return both text and audio response
                return jsonify({
                    "text": ai_response,
                    "audio_url": f"/api/audio?path={tmp_mp3.name}"
                })
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio')
def serve_audio():
    path = request.args.get('path')
    return send_file(path, mimetype='audio/mpeg')

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
