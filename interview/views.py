import os
import platform
from django.shortcuts import render
import speech_recognition as sr
import requests
import json

# Create your views here.

if platform.system() == "Darwin":  # macOS
    sr.AudioFile.FLAC_converter = 'flac'
    sr.Recognizer.FLAC_converter = 'flac'  # Also override recognizer default
elif platform.system() == "Windows":
    sr.AudioFile.FLAC_converter = 'flac.exe'
    sr.Recognizer.FLAC_converter = 'flac.exe'

def ask_question():
    question = "Tell me about yourself."
    print("\n🧑‍💼 Interviewer: " + question)
    return question

def transcribe_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        input("\n🎤 Press [ENTER] to start recording...")
        print("🎙️ Listening... (Press [CTRL+C] to force stop if needed)")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)  # No time limit!

    try:
        print("📝 Transcribing...")
        text = recognizer.recognize_google(audio)
        print("📄 Transcript:", text)
        return text
    except Exception as e:
        print("❌ Error in transcription:", str(e))
        return None
    
def get_ollama_feedback(user_answer, question):
    print("🤖 Sending to Ollama LLM...")
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "mistral",
        "stream": False,        
        "messages": [
            {"role": "system", "content": "You are an empathetic interview coach. Give helpful and constructive feedback on mock interview answers."},
            {"role": "user", "content": f"Interview Question: {question}\nAnswer: {user_answer}\n\nPlease give honest, constructive feedback like a career coach."}
        ]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        reply = response.json()['message']['content']
        print("\n✅ GPT Feedback:\n", reply)
    except Exception as e:
        print("❌ Error from Ollama:", str(e))

def mock_interview_view(request):
    if request.method == 'POST':
        question = ask_question()
        try:
            answer = transcribe_speech()
            feedback = get_ollama_feedback(answer, question)
            return render(request, 'interview.html', {
                'question': question,
                'answer': answer,
                'feedback': feedback
            })
        except Exception as e:
            return HttpResponse("Error: " + str(e))
    return render(request, 'interview.html')
