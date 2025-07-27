# Enhanced views.py with camera fallback and better Windows compatibility
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
import time
import re
import json
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from deepface import DeepFace
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Camera configuration for Windows
def configure_camera_for_windows():
    """Configure camera settings specifically for Windows to avoid MSMF errors"""
    if platform.system() == "Windows":
        # Set environment variable to use DirectShow instead of MSMF
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1"

# Configure camera on module load
configure_camera_for_windows()

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
        "What would you do if you realized your client misunderstood the software's capabilities after you delivered the project?",
    ],
    'technical': [
        "You're asked to design a system for handling real-time data. How would you approach it?",
        "You notice a bug in your teammate's code that could cause data loss. What do you do?",
        "You're deploying an app and the production server crashes. Walk me through your response.",
        "You're assigned to maintain an unfamiliar legacy codebase with no documentation. How do you begin?",
        "You need to optimize a feature that's causing performance issues. How do you identify and resolve the bottleneck?",
    ],
    'motivational': [
        "Why do you want this role?",
        "Tell me about a time you were proud of your work. What made it meaningful?",
        "What motivates you to keep going when the work gets tough or repetitive?",
        "What kind of impact do you want to make in this field?",
        "Describe your ideal work environment and why it brings out your best.",
    ]
}

# Global variables for emotion detection
stop_event = threading.Event()
detected_emotion = "neutral"
emotion_thread = None
camera_available = False
emotion_detection_enabled = True

def try_multiple_camera_backends():
    """Try different camera backends to find one that works"""
    backends = []
    
    if platform.system() == "Windows":
        backends = [
            cv2.CAP_DSHOW,    # DirectShow (usually more stable on Windows)
            cv2.CAP_MSMF,     # Microsoft Media Foundation
            cv2.CAP_V4L2,     # Video4Linux2 (if available)
            cv2.CAP_ANY       # Let OpenCV choose
        ]
    else:
        backends = [
            cv2.CAP_V4L2,     # Video4Linux2 (Linux)
            cv2.CAP_AVFOUNDATION, # macOS
            cv2.CAP_ANY       # Let OpenCV choose
        ]
    
    for backend in backends:
        try:
            logger.info(f"Trying camera backend: {backend}")
            cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                # Test if we can actually read frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info(f"Successfully initialized camera with backend: {backend}")
                    cap.release()
                    return backend
                else:
                    logger.warning(f"Backend {backend} opened but cannot read frames")
            
            cap.release()
            
        except Exception as e:
            logger.warning(f"Backend {backend} failed: {e}")
            continue
    
    logger.error("No working camera backend found")
    return None

def check_camera_availability():
    """Enhanced camera availability check with multiple backends"""
    global camera_available
    
    try:
        working_backend = try_multiple_camera_backends()
        if working_backend is not None:
            camera_available = True
            logger.info("Camera is available and working")
            return working_backend
        else:
            camera_available = False
            logger.warning("No working camera found")
            return None
            
    except Exception as e:
        camera_available = False
        logger.error(f"Camera availability check failed: {e}")
        return None

def detect_emotion_live():
    """Enhanced emotion detection with multiple backend support"""
    global detected_emotion, camera_available
    
    working_backend = check_camera_availability()
    if working_backend is None:
        logger.warning("Skipping emotion detection - no working camera backend")
        detected_emotion = "neutral"
        return
    
    cap = None
    try:
        cap = cv2.VideoCapture(0, working_backend)
        if not cap.isOpened():
            logger.warning("Could not open camera for emotion detection")
            detected_emotion = "neutral"
            return
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 5)  # Lower FPS to reduce load
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower resolution
        
        consecutive_failures = 0
        max_failures = 3  # Reduced threshold
        emotion_update_counter = 0
        
        logger.info("Starting emotion detection loop")
        
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed (attempt {consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.warning("Too many camera read failures, stopping emotion detection")
                        break
                    time.sleep(0.5)
                    continue
                
                consecutive_failures = 0
                
                # Only analyze emotion every few frames to reduce load
                emotion_update_counter += 1
                if emotion_update_counter % 5 == 0:  # Update every 5th frame
                    try:
                        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        if result and len(result) > 0:
                            new_emotion = result[0].get('dominant_emotion', 'neutral')
                            if new_emotion != detected_emotion:
                                detected_emotion = new_emotion
                                logger.info(f"Emotion updated to: {detected_emotion}")
                    except Exception as e:
                        logger.warning(f"DeepFace analysis error: {e}")
                        # Don't increment consecutive_failures for DeepFace errors
                
                # Longer delay to reduce system load
                time.sleep(0.2)
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Emotion detection frame error: {e}")
                if consecutive_failures >= max_failures:
                    logger.warning("Too many consecutive failures, stopping emotion detection")
                    break
                time.sleep(0.5)
                continue
    
    except Exception as e:
        logger.error(f"Emotion detection setup error: {e}")
        detected_emotion = "neutral"
    
    finally:
        if cap is not None:
            try:
                cap.release()
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
    
    logger.info("Emotion detection thread ended")

def start_page(request):
    return render(request, 'start.html')

def interview_page(request):
    global stop_event, emotion_thread, emotion_detection_enabled
    
    category = request.GET.get('category', 'behavioral')
    questions = interview_questions.get(category, interview_questions['behavioral'])
    question = random.choice(questions)
    request.session['question'] = question
    request.session['category'] = category

    # Check if user wants to disable emotion detection
    disable_camera = request.GET.get('disable_camera', '').lower() == 'true'
    emotion_detection_enabled = not disable_camera

    # Stop any existing emotion detection
    stop_event.set()
    if emotion_thread and emotion_thread.is_alive():
        emotion_thread.join(timeout=3)
    
    # Start new emotion detection only if enabled
    if emotion_detection_enabled:
        stop_event.clear()
        emotion_thread = threading.Thread(target=detect_emotion_live)
        emotion_thread.daemon = True
        emotion_thread.start()
        logger.info("Started emotion detection thread")
    else:
        logger.info("Emotion detection disabled by user")

    return render(request, 'interview.html', {
        'question': question,
        'camera_available': camera_available,
        'emotion_detection_enabled': emotion_detection_enabled
    })

# Add a new endpoint to manually set emotion
@csrf_exempt
def set_emotion(request):
    """Allow manual emotion setting if camera is not available"""
    global detected_emotion
    
    if request.method == "POST":
        emotion = request.POST.get('emotion', 'neutral')
        valid_emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
        
        if emotion in valid_emotions:
            detected_emotion = emotion
            logger.info(f"Emotion manually set to: {emotion}")
            return JsonResponse({'success': True, 'emotion': emotion})
        else:
            return JsonResponse({'error': 'Invalid emotion'})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def transcribe_audio(request):
    if request.method == "POST":
        temp_webm_path = None
        wav_path = None
        
        try:
            audio_data = request.POST.get('audio_data', '')
            logger.info(f"Received audio_data length: {len(audio_data)}")

            if audio_data.startswith("data:audio/webm;base64,"):
                audio_data = audio_data.split(",")[1]

            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"Decoded audio data size (bytes): {len(audio_bytes)}")

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

            logger.info(f"Transcription successful: {transcript[:50]}...")
            return JsonResponse({'transcript': transcript})
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return JsonResponse({'error': 'Could not understand the audio. Please speak clearly and try again.'})
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return JsonResponse({'error': 'Speech recognition service unavailable. Please try again later.'})
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return JsonResponse({'error': f'Transcription failed: {str(e)}'})
        finally:
            # Clean up temporary files
            for file_path in [temp_webm_path, wav_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {file_path}: {e}")
    
    return JsonResponse({'error': 'Invalid request method'})


def analyze_star_framework(answer):
    """Analyze how well the answer follows STAR framework"""
    answer_lower = answer.lower()
    
    situation_words = ['when', 'time', 'situation', 'project', 'experience', 'while', 'during']
    task_words = ['needed', 'had to', 'responsible', 'task', 'goal', 'objective']
    action_words = ['i did', 'i took', 'i decided', 'i implemented', 'i created', 'i organized']
    result_words = ['result', 'outcome', 'achieved', 'improved', 'increased', 'successful']
    
    situation_score = sum(1 for word in situation_words if word in answer_lower)
    task_score = sum(1 for word in task_words if word in answer_lower)
    action_score = sum(1 for word in action_words if word in answer_lower)
    result_score = sum(1 for word in result_words if word in answer_lower)
    
    missing = []
    if situation_score == 0: missing.append("Situation context")
    if task_score == 0: missing.append("Task/responsibility")
    if action_score == 0: missing.append("Specific actions")
    if result_score == 0: missing.append("Results/outcomes")
    
    if not missing:
        return "Excellent! You covered all STAR elements (Situation, Task, Action, Result)."
    elif len(missing) == 1:
        return f"Good structure! Consider adding more detail about: {missing[0]}."
    else:
        return f"Consider strengthening these STAR elements: {', '.join(missing)}."

def analyze_problem_solving(answer):
    """Analyze problem-solving approach in situational questions"""
    answer_lower = answer.lower()
    
    understanding_words = ['understand', 'analyze', 'assess', 'evaluate', 'consider']
    options_words = ['option', 'alternative', 'approach', 'solution', 'choice']
    reasoning_words = ['because', 'since', 'therefore', 'reason', 'due to']
    
    understanding = any(word in answer_lower for word in understanding_words)
    options = any(word in answer_lower for word in options_words)
    reasoning = any(word in answer_lower for word in reasoning_words)
    
    if understanding and options and reasoning:
        return "Great problem-solving structure! You showed understanding, considered options, and provided reasoning."
    else:
        missing = []
        if not understanding: missing.append("situation analysis")
        if not options: missing.append("alternative approaches")
        if not reasoning: missing.append("clear reasoning")
        return f"Good start! Consider adding more about: {', '.join(missing)}."

def analyze_technical_approach(answer):
    """Analyze technical thinking"""
    answer_lower = answer.lower()
    
    technical_words = ['system', 'design', 'implement', 'architecture', 'performance', 'scale']
    process_words = ['first', 'then', 'next', 'finally', 'step']
    consideration_words = ['trade-off', 'consider', 'balance', 'optimize', 'constraint']
    
    technical = any(word in answer_lower for word in technical_words)
    process = any(word in answer_lower for word in process_words)
    considerations = any(word in answer_lower for word in consideration_words)
    
    strengths = []
    if technical: strengths.append("technical terminology")
    if process: strengths.append("systematic approach")
    if considerations: strengths.append("trade-off awareness")
    
    if len(strengths) >= 2:
        return f"Strong technical thinking! You demonstrated {' and '.join(strengths)}."
    else:
        return "Good technical foundation. Consider adding more systematic thinking and trade-off discussions."

def analyze_motivation(answer):
    """Analyze motivational elements"""
    answer_lower = answer.lower()
    
    passion_words = ['passionate', 'excited', 'love', 'enjoy', 'motivate']
    values_words = ['value', 'important', 'believe', 'principle']
    growth_words = ['learn', 'grow', 'develop', 'improve', 'challenge']
    
    passion = any(word in answer_lower for word in passion_words)
    values = any(word in answer_lower for word in values_words)
    growth = any(word in answer_lower for word in growth_words)
    
    if passion and values and growth:
        return "Excellent! You showed genuine passion, clear values, and growth mindset."
    else:
        suggestions = []
        if not passion: suggestions.append("express more enthusiasm")
        if not values: suggestions.append("connect to your values")
        if not growth: suggestions.append("mention learning/growth goals")
        return f"Good foundation! Consider: {', '.join(suggestions)}."

def get_emotion_feedback(emotion):
    """Provide feedback based on detected emotion"""
    emotion_feedback = {
        'neutral': "You maintained a calm, professional demeanor throughout your response.",
        'happy': "Your positive energy came through well! This enthusiasm is great for interviews.",
        'sad': "You seemed a bit subdued. Try to project more energy and confidence in your delivery.",
        'angry': "You appeared tense. Take deep breaths and focus on staying calm and composed.",
        'fear': "Some nervousness is normal! Practice will help build confidence. Focus on your achievements.",
        'surprise': "You seemed caught off-guard. Take a moment to collect your thoughts before answering.",
        'disgust': "You appeared uncomfortable. Try to maintain a more positive, engaged expression."
    }

def generate_example_improvement(answer, category):
    """Generate a specific example of how to improve part of the answer"""
    
    examples = {
        'behavioral': "Instead of 'I worked on a project', try 'I led a 5-person team to deliver a customer portal that increased user satisfaction by 30% over 3 months.'",
        'situational': "Instead of 'I would talk to them', try 'I would schedule a private conversation to understand their perspective, then work together on a solution that addresses both our concerns.'",
        'technical': "Instead of 'I would fix the bug', try 'I would first reproduce the issue, analyze the root cause using debugging tools, implement a targeted fix, and add unit tests to prevent regression.'",
        'motivational': "Instead of 'I want to grow', try 'I'm excited about this role because it combines my passion for user experience with my goal of leading product strategy in a data-driven environment.'"
    }
    
    return examples.get(category, "Add more specific details and concrete examples to make your answer more compelling.")

# OLLAMA INTEGRATION FUNCTIONS
def get_ollama_feedback(user_answer, question, emotion, category="behavioral"):
    """Optimized feedback function with shorter, faster prompts"""
    logger.info("Submitting response to LLM for feedback...")
    logger.info(f"Detected Emotion: {emotion}")

    # Simplified category frameworks - much shorter
    frameworks = {
        'behavioral': 'STAR: Situation, Task, Action, Result',
        'situational': 'Problem-solving: Understand, Options, Choice, Reasoning',
        'technical': 'Technical: Problem, Approach, Implementation, Trade-offs',
        'motivational': 'Values: Motivation, Alignment, Knowledge, Vision'
    }

    framework = frameworks.get(category, frameworks['behavioral'])

    url = "http://localhost:11434/api/chat"
    
    # Much shorter, focused prompt
    payload = {
        "model": "mistral",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": f"You are an interview coach. Evaluate {category} interview answers using {framework}. Be encouraging but specific about improvements. Keep feedback concise and actionable."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {user_answer}\nEmotion: {emotion}\n\nProvide brief feedback on:\n1. Framework usage\n2. Strengths\n3. Missing elements\n4. Specific improvements\n5. Emotion impact"
            }
        ]
    }

    # Enhanced connection handling
    max_retries = 2  # Reduced retries
    timeout = 45     # Reduced timeout
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Ollama (attempt {attempt + 1}/{max_retries})")
            
            # Check if Ollama is responsive first
            try:
                health_check = requests.get("http://localhost:11434/api/tags", timeout=5)
                if health_check.status_code != 200:
                    logger.warning("Ollama health check failed")
                    return "Ollama service is not responding properly. Please restart Ollama and try again."
            except requests.exceptions.RequestException:
                logger.warning("Cannot reach Ollama service")
                return "Cannot connect to Ollama. Please ensure it's running with 'ollama serve' and try again."
            
            # Make the actual request
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                logger.info("Feedback received successfully")
                return result['message']['content']
            else:
                logger.warning(f"Unexpected response format: {result}")
                return "Received response but format was unexpected. Please try again."
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return f"Request timed out after {timeout} seconds. Your answer might be too long, or Ollama is overloaded. Try a shorter response or restart Ollama."
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return "Connection failed. Please check that Ollama is running with 'ollama serve' and try again."
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            if attempt == max_retries - 1:
                return f"Server error ({e.response.status_code}). Please try again or restart Ollama."
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                return f"Unexpected error occurred: {str(e)}"
        
        # Short wait before retry
        if attempt < max_retries - 1:
            time.sleep(2)

    return "Failed to get feedback. Please ensure Ollama is running and try again."

# Alternative: Even more minimal version if still timing out
def get_minimal_ollama_feedback(user_answer, question, emotion, category="behavioral"):
    """Ultra-minimal version for faster processing"""
    logger.info("Using minimal feedback mode")
    
    url = "http://localhost:11434/api/chat"
    
    # Extremely short prompt
    payload = {
        "model": "mistral",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": f"As an interview coach, briefly evaluate this {category} interview answer:\n\nQ: {question}\nA: {user_answer}\n\nGive 3 quick points: what's good, what's missing, how to improve."
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['message']['content']
    except Exception as e:
        return f"Quick feedback unavailable: {str(e)}"

# Function to check Ollama model status
def check_ollama_status():
    """Check if Ollama is running and which models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            logger.info(f"Ollama is running. Available models: {model_names}")
            return True, model_names
        else:
            logger.warning("Ollama responded but with error status")
            return False, []
    except Exception as e:
        logger.error(f"Cannot reach Ollama: {e}")
        return False, []

def get_optimized_ollama_feedback(user_answer, question, emotion, category="behavioral"):
    """Optimized Ollama feedback with shorter prompts and better error handling"""
    
    # Category-specific short frameworks
    frameworks = {
        'behavioral': 'Use STAR method: Situation, Task, Action, Result',
        'situational': 'Show: Understanding, Options, Decision, Reasoning', 
        'technical': 'Cover: Problem, Approach, Implementation, Trade-offs',
        'motivational': 'Express: Passion, Values, Growth mindset, Alignment'
    }
    
    framework = frameworks.get(category, frameworks['behavioral'])
    
    # Much shorter, focused prompt
    prompt = f"""Question: {question}

Answer: {user_answer}

Emotion detected: {emotion}

As an interview coach, provide concise feedback on this {category} interview answer. {framework}.

Give 4 brief points:
1. What's working well
2. What's missing or weak  
3. How to improve
4. Emotion/delivery notes

Keep response under 200 words."""

    payload = {
        "model": "mistral",
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful interview coach. Be encouraging but specific. Keep feedback concise and actionable."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        logger.info("Sending optimized request to Ollama...")
        response = requests.post(
            "http://localhost:11434/api/chat", 
            json=payload, 
            timeout=25  # Shorter timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                feedback = result['message']['content'].strip()
                logger.info("AI feedback received successfully")
                return feedback
            else:
                logger.warning("Invalid response format from Ollama")
                return None
        else:
            logger.warning(f"Ollama returned status code: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Connection to Ollama failed")
        return None
    except Exception as e:
        logger.error(f"Ollama request error: {e}")
        return None

# Complete the missing parts from the first views.py file

def get_smart_feedback_with_fallback(user_answer, question, emotion, category="behavioral"):
    """Enhanced feedback with better error handling and fallback"""
    
    # First, check if Ollama is available with a quick health check
    try:
        logger.info("Testing Ollama connection...")
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            
            # Check if mistral model is available
            if any('mistral' in str(model) for model in models):
                logger.info("Ollama and Mistral model available - attempting AI feedback")
                
                # Try to get AI feedback with shorter timeout
                try:
                    ai_feedback = get_optimized_ollama_feedback(user_answer, question, emotion, category)
                    if ai_feedback and not ai_feedback.startswith("Failed") and not ai_feedback.startswith("Request timed out"):
                        return "🤖 **AI Feedback:**\n\n" + ai_feedback
                    else:
                        logger.warning("AI feedback failed, using fallback")
                except Exception as e:
                    logger.warning(f"AI feedback error: {e}")
            else:
                logger.warning("Mistral model not found in Ollama")
        else:
            logger.warning(f"Ollama health check failed: {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.warning("Ollama connection timeout - using fallback")
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama - using fallback")
    except Exception as e:
        logger.warning(f"Ollama check failed: {e}")
    
    # Fallback to rule-based system
    logger.info("Using rule-based feedback system")
    return "💡 **Smart Analysis:**\n\n" + get_fallback_feedback(user_answer, question, emotion, category)

# FALLBACK FEEDBACK SYSTEM FUNCTIONS
def get_fallback_feedback(user_answer, question, emotion, category="behavioral"):
    """Provide rule-based feedback when Ollama is unavailable"""
    
    # Analyze answer characteristics
    word_count = len(user_answer.split())
    sentence_count = len([s for s in user_answer.split('.') if s.strip()])
    
    # Check for framework elements based on category
    feedback_parts = []
    
    # General structure analysis
    if word_count < 50:
        feedback_parts.append("📏 **Length**: Your answer is quite brief. Consider adding more specific details and examples to strengthen your response.")
    elif word_count > 200:
        feedback_parts.append("📏 **Length**: Good detail level! Make sure to stay focused on the key points to keep the interviewer engaged.")
    else:
        feedback_parts.append("📏 **Length**: Good answer length - detailed but concise.")
    
    # Category-specific analysis
    if category == 'behavioral':
        star_elements = analyze_star_framework(user_answer)
        feedback_parts.append(f"⭐ **STAR Framework**: {star_elements}")
        
    elif category == 'situational':
        problem_solving = analyze_problem_solving(user_answer)
        feedback_parts.append(f"🧠 **Problem-Solving**: {problem_solving}")
        
    elif category == 'technical':
        technical_elements = analyze_technical_approach(user_answer)
        feedback_parts.append(f"🔧 **Technical Approach**: {technical_elements}")
        
    elif category == 'motivational':
        motivation_elements = analyze_motivation(user_answer)
        feedback_parts.append(f"💪 **Motivation**: {motivation_elements}")
    
    # Emotion-based feedback
    emotion_feedback = get_emotion_feedback(emotion)
    feedback_parts.append(f"😊 **Emotional Delivery**: {emotion_feedback}")
    
    # Specific improvements
    improvements = generate_improvements(user_answer, category, word_count)
    feedback_parts.append(f"🎯 **Key Improvements**: {improvements}")
    
    # Example enhancement
    example = generate_example_improvement(user_answer, category)
    if example:
        feedback_parts.append(f"💡 **Example Enhancement**: {example}")
    
    return "\n\n".join(feedback_parts)

def generate_improvements(answer, category, word_count):
    """Generate specific improvement suggestions"""
    improvements = []
    
    if 'I' not in answer:
        improvements.append("Use more 'I' statements to show personal ownership and responsibility")
    
    if not any(char.isdigit() for char in answer):
        improvements.append("Add specific metrics, numbers, or timeframes to quantify your impact")
    
    if word_count < 30:
        improvements.append("Expand with more specific details and concrete examples")
    
    if category == 'behavioral' and 'learned' not in answer.lower():
        improvements.append("Mention what you learned or how you'd handle similar situations differently")
    
    if not improvements:
        improvements.append("Practice delivering your answer with more confidence and energy")
    
    return "; ".join(improvements[:3])  # Limit to top 3

# ADD this endpoint for testing Ollama status (optional):
@csrf_exempt
def check_ollama_status_endpoint(request):
    """Endpoint to check Ollama status from frontend"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            has_mistral = any('mistral' in str(model) for model in models)
            
            return JsonResponse({
                'status': 'running',
                'models': [m.get('name', str(m)) for m in models],
                'has_mistral': has_mistral
            })
        else:
            return JsonResponse({'status': 'error', 'message': 'Ollama not responding properly'})
            
    except Exception as e:
        return JsonResponse({'status': 'offline', 'error': str(e)})

# ENHANCE your existing feedback_page function by replacing it with this:
def feedback_page(request):
    global stop_event, emotion_thread
    
    # Stop emotion detection
    stop_event.set()
    if emotion_thread and emotion_thread.is_alive():
        emotion_thread.join(timeout=5)
    
    question = request.session.get('question', '')
    answer = request.GET.get('transcript') or request.session.get('answer', '')
    request.session['answer'] = answer
    emotion = detected_emotion
    category = request.session.get('category', 'behavioral')

    # Use enhanced feedback system
    logger.info(f"Generating feedback for {category} question")
    logger.info(f"Answer length: {len(answer.split())} words")
    logger.info(f"Detected emotion: {emotion}")
    
    start_time = time.time()
    feedback = get_smart_feedback_with_fallback(answer, question, emotion, category)
    end_time = time.time()
    
    logger.info(f"Feedback generated in {end_time - start_time:.2f} seconds")

    # Check if we used AI or fallback
    feedback_type = "AI" if feedback.startswith("🤖") else "Rule-based"
    
    return render(request, 'feedback.html', {
        'question': question,
        'answer': answer,
        'emotion': emotion,
        'feedback': feedback,
        'feedback_type': feedback_type,
        'category': category,
        'camera_available': camera_available,
        'processing_time': f"{end_time - start_time:.1f}s"
    })