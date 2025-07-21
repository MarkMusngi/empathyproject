# views.py
import os
import cv2
import tempfile
import platform
import io
import wave
import base64
import logging
import threading
import random
import requests
import speech_recognition as sr
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from deepface import DeepFace
from pydub import AudioSegment

logger = logging.getLogger(__name__)

interview_questions = [
    "Tell me about yourself.",
    "What are your greatest strengths?",
    "What are your weaknesses?",
    "Why do you want to work here?",
    "Where do you see yourself in five years?",
    "Tell me about a challenge you faced and how you handled it.",
    "Why should we hire you?",
    "Describe a time you worked in a team.",
    "What do you know about our company?",
    "How do you handle pressure and stress?"
]

stop_event = threading.Event()
detected_emotion = "neutral"

# Emotion detection thread
def detect_emotion_live():
    global detected_emotion
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']
        except Exception as e:
            logger.warning(f"Emotion detection error: {e}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Start page

def start_page(request):
    return render(request, 'start.html')

# Interview page

def interview_page(request):
    question = random.choice(interview_questions)
    request.session['question'] = question

    global stop_event
    stop_event.clear()
    emotion_thread = threading.Thread(target=detect_emotion_live)
    emotion_thread.start()

    return render(request, 'interview.html', {'question': question})

# Audio transcription endpoint

@csrf_exempt
def transcribe_audio(request):
    if request.method == "POST":
        try:
            audio_data = request.POST.get('audio_data', '')
            print("Received audio_data length:", len(audio_data))

            if audio_data.startswith("data:audio/webm;base64,"):
                audio_data = audio_data.split(",")[1]

            audio_bytes = base64.b64decode(audio_data)
            print("Decoded audio data size (bytes):", len(audio_bytes))

            # Save the WebM data temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
                temp_webm.write(audio_bytes)
                temp_webm_path = temp_webm.name

            # Convert to WAV using pydub
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                audio.export(temp_wav.name, format="wav")
                wav_path = temp_wav.name

            # Transcribe using SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)

            # Store transcript in session
            answer = request.GET.get('transcript') or request.session.get('answer', '')
            request.session['answer'] = answer

            # Clean up
            os.remove(temp_webm_path)
            os.remove(wav_path)

            return JsonResponse({'transcript': transcript})
        except Exception as e:
            print("Error during transcription:", str(e))
            return JsonResponse({'error': f'Transcription failed: {str(e)}'})
    return JsonResponse({'error': 'Invalid request method'})

def get_ollama_feedback(user_answer, question, emotion):
    print("Submitting response to LLM for feedback...")
    print(f"Detected Emotion: {emotion}")

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "mistral",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an empathetic interview coach. "
                    "Your role is to give constructive and supportive feedback on mock interview responses. "
                    "Please also consider the speaker's detected emotional state (e.g., happy, neutral, sad, angry) "
                    "when offering your feedback. "
                    "If the speaker seems nervous or unconfident, give encouragement and suggest improvements. "
                    "If they seem confident or enthusiastic, reinforce that positively."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Interview Question: {question}\n"
                    f"Answer: {user_answer}\n"
                    f"Detected Emotion: {emotion}\n\n"
                    "Please:\n"
                    "- Comment on the quality of the answer\n"
                    "- Comment on the detected emotion\n"
                    "- Give constructive feedback as a career coach\n"
                    "- Keep it helpful, supportive, and professional"
                )
            }
        ]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        return f"Error receiving feedback: {e}"

def feedback_page(request):
    stop_event.set()
    question = request.session.get('question', '')
    answer = request.GET.get('transcript') or request.session.get('answer', '')
    request.session['answer'] = answer  # store for reuse if needed
    emotion = detected_emotion

    feedback = get_ollama_feedback(answer, question, emotion)

    return render(request, 'feedback.html', {
        'question': question,
        'answer': answer,
        'emotion': emotion,
        'feedback': feedback
    })
