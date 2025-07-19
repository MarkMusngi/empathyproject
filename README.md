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
