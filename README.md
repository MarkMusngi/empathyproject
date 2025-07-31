# Interview Feedback Simulator

This project simulates a mock interview session using speech input, real-time emotion detection, and local LLM-based feedback. It is designed to help users practice and receive constructive interview feedback.

## Features

- Microphone-based voice recording
- Emotion detection via webcam using DeepFace
- Real-time transcription using Google Speech Recognition
- Feedback generation using a local LLM model via Ollama

## Prerequisites

Ensure you have the following installed:

### Python

- Python 3.8 or higher

### System Tools

- Microphone and webcam
- FLAC encoder (used by the SpeechRecognition library)
  - Windows: [flac.exe](https://xiph.org/flac/download.html) and add to your PATH
  - macOS: Use `brew install flac`

### Ollama (for local LLM)

1. Download Ollama from https://ollama.com/
2. Install and start it (typically runs in the background)
3. Pull the desired model:
   ```bash
   ollama pull mistral

pip install opencv-python deepface SpeechRecognition requests pyaudio


# Core Django and web framework
pip install django
pip install djangorestframework  # If using REST APIs

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install numpy scipy
pip install tensorflow
pip install django
pip install opencv-python
pip install deepface
pip install SpeechRecognition
pip install pyaudio
pip install pydub
pip install sentence-transformers
pip install scikit-learn
pip install vaderSentiment
pip install requests

# Run Django migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Start development server
python manage.py runserver

python manage.py runserver 127.0.0.1:8001