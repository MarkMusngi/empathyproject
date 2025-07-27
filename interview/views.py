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

interview_questions = {
    'behavioral': [
        "Tell me about a time you had to work with a difficult colleague or team member. How did you handle it?",
        "Describe a time when you had to make a quick decision with incomplete information.",
        "Give an example of a project you led. What challenges did you face and how did you overcome them?",
        "Talk about a time you failed. What did you learn from it?",
        "Tell me about a time you had to adapt to a big change at work or school.",
    ],
    'situational': [
        "What would you do if you saw a teammate taking credit for your work?",
        "If you were given two urgent tasks by two supervisors, how would you prioritize?",
        "How would you handle a situation where a key team member is underperforming close to a deadline?",
        "Imagine you're presenting to a group and someone challenges your data in front of everyone. How would you respond?",
        "What would you do if you realized your client misunderstood the software’s capabilities after you delivered the project?",
    ],
    'technical': [
        "You’re asked to design a system for handling real-time data. How would you approach it?",
        "You notice a bug in your teammate’s code that could cause data loss. What do you do?",
        "You’re deploying an app and the production server crashes. Walk me through your response.",
        "You’re assigned to maintain an unfamiliar legacy codebase with no documentation. How do you begin?",
        "You need to optimize a feature that’s causing performance issues. How do you identify and resolve the bottleneck?",
    ],
    'motivational': [
        "Why do you want this role?",
        "Tell me about a time you were proud of your work. What made it meaningful?",
        "What motivates you to keep going when the work gets tough or repetitive?",
        "What kind of impact do you want to make in this field?",
        "Describe your ideal work environment and why it brings out your best.",
    ]
}

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
    category = request.GET.get('category', 'behavioral')  # Default to behavioral
    questions = interview_questions.get(category, interview_questions['behavioral'])
    question = random.choice(questions)
    request.session['question'] = question
    request.session['category'] = category

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
