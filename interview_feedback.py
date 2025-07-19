import os
import platform
import cv2
import speech_recognition as sr
import requests
import threading
import ctypes
import random
import time
from deepface import DeepFace

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

# FLAC converter setup
if platform.system() == "Darwin":
    sr.AudioFile.FLAC_converter = 'flac'
    sr.Recognizer.FLAC_converter = 'flac'
elif platform.system() == "Windows":
    sr.AudioFile.FLAC_converter = 'flac.exe'
    sr.Recognizer.FLAC_converter = 'flac.exe'

def detect_emotion_live():
    global detected_emotion
    cap = cv2.VideoCapture(0)

    print("Emotion detection started")

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            detected_emotion = dominant_emotion

            cv2.putText(frame, f"Emotion: {dominant_emotion}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detection ended.")

def transcribe_until_enter():
    import io
    import wave

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("\nPress [ENTER] to begin recording.")
    input()

    print("Calibrating microphone. Please remain silent...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("Recording in progress. Press [ENTER] to stop.")
        print("Recording...")

        audio_frames = []

        def record_loop():
            while not stop_event.is_set():
                try:
                    frame = source.stream.read(source.CHUNK)
                    audio_frames.append(frame)
                except Exception as e:
                    print("Audio capture error:", e)
                    break

        stop_event.clear()
        record_thread = threading.Thread(target=record_loop)
        record_thread.start()

        input()  # Wait for user to press ENTER
        stop_event.set()
        record_thread.join()
        print("Recording completed.")

        # Save to memory buffer for recognition
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(source.SAMPLE_WIDTH)
            wf.setframerate(source.SAMPLE_RATE)
            wf.writeframes(b''.join(audio_frames))

        wav_buffer.seek(0)
        audio = sr.AudioFile(wav_buffer)

        with audio as source:
            audio_data = recognizer.record(source)

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("Transcript:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print("API request error:", str(e))
    return None

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
        reply = response.json()['message']['content']
        print("\nFeedback:\n", reply)
    except Exception as e:
        print("Error receiving feedback:", str(e))

def clean_exit():
    print("\nInterview simulation completed. Exiting.")
    pid = os.getpid()
    ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.OpenProcess(1, False, pid), 0)

if __name__ == "__main__":
    selected_question = random.choice(interview_questions)

    emotion_thread = threading.Thread(target=detect_emotion_live)
    emotion_thread.daemon = True
    emotion_thread.start()
    time.sleep(10)
    print(f"\nInterviewer: {selected_question}")

    answer = transcribe_until_enter()

    stop_event.set()
    emotion_thread.join()

    if answer:
        get_ollama_feedback(answer, selected_question, detected_emotion)

    clean_exit()
