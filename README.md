# Mock Interview Assistant

This project is a local mock interview assistant built with Python, Speech Recognition, and a local Large Language Model (LLM) using [Ollama](https://ollama.com). It simulates an interview scenario by:

- Asking an interview question
- Capturing and transcribing the user’s spoken answer
- Sending the answer to an LLM (e.g., Mistral) running locally via Ollama
- Returning constructive feedback based on the user's response

---

## Features

-  Voice-based input using your microphone
-  Real-time transcription using Google's Web Speech API
-  Local LLM-powered feedback using Ollama (no OpenAI API required)
-  Easy to set up and extend

---

## Requirements

### System Requirements

- Python 3.8+
- A functioning microphone
- Internet connection (for Google Speech API transcription only)
- Ollama installed locally with a supported model (e.g., Mistral)

---

## Setup Instructions

### 1. Install Python Dependencies

- Install the required Python libraries using pip:

- ```bash
- pip install speechrecognition pyaudio requests


### 2. Install Ollama (Local LLM)
- ollama run mistral

### 3. Run the python script
- python interview_feedback.py

## Run Web app
- python manage.py runserver

wthelly
