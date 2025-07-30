# Enhanced views.py with vectorizer integration and cleanup
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
from collections import Counter
from .emotion_validator import get_emotion_validator, EmotionValidation


# Import the content vectorizer
from .content_vectorizer import ContentVectorizer, SemanticScore


logger = logging.getLogger(__name__)

# Global vectorizer instance - initialize once when Django starts
_vectorizer_instance = None
validated_emotion = "neutral"
emotion_confidence = 1.0
current_answer_text = ""


def get_vectorizer():
    """Singleton pattern for vectorizer - initialize once, use throughout app"""
    global _vectorizer_instance
    if _vectorizer_instance is None:
        try:
            # Use Django's media/cache directory for vectors
            cache_dir = os.path.join(os.path.dirname(__file__), 'vector_cache')
            _vectorizer_instance = ContentVectorizer(cache_dir=cache_dir)
            logger.info("Content vectorizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vectorizer: {e}")
            _vectorizer_instance = None
    return _vectorizer_instance

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
    """Enhanced emotion detection with real-time validation"""
    global detected_emotion, validated_emotion, emotion_confidence, camera_available, current_answer_text
    
    working_backend = check_camera_availability()
    if working_backend is None:
        logger.warning("Skipping emotion detection - no working camera backend")
        detected_emotion = "neutral"
        validated_emotion = "neutral"
        return
    
    cap = None
    validator = get_emotion_validator()
    
    try:
        cap = cv2.VideoCapture(0, working_backend)
        if not cap.isOpened():
            logger.warning("Could not open camera for emotion detection")
            detected_emotion = "neutral"
            validated_emotion = "neutral"
            return
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 5)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        consecutive_failures = 0
        max_failures = 3
        emotion_update_counter = 0
        last_validation_time = 0
        
        logger.info("Starting enhanced emotion detection with validation")
        
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
                if emotion_update_counter % 5 == 0:
                    try:
                        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        if result and len(result) > 0:
                            raw_emotion = result[0].get('dominant_emotion', 'neutral')
                            
                            # Update detected emotion
                            if raw_emotion != detected_emotion:
                                detected_emotion = raw_emotion
                                logger.info(f"Raw emotion detected: {detected_emotion}")
                            
                            # Validate emotion every 3 seconds if we have text
                            current_time = time.time()
                            if (validator and current_answer_text and 
                                current_time - last_validation_time > 3.0):
                                
                                validation = validator.validate_emotion(
                                    detected_emotion, current_answer_text
                                )
                                
                                validated_emotion = validation.validated_emotion
                                emotion_confidence = validation.confidence_score
                                
                                if validation.flags:
                                    logger.info(f"Validation flags: {validation.flags}")
                                
                                if detected_emotion != validated_emotion:
                                    logger.info(f"Emotion adjusted: {detected_emotion} -> {validated_emotion} "
                                              f"(confidence: {emotion_confidence:.3f})")
                                
                                last_validation_time = current_time
                            else:
                                # No validation yet, use raw emotion
                                validated_emotion = detected_emotion
                                emotion_confidence = 0.7  # Default confidence
                                
                    except Exception as e:
                        logger.warning(f"DeepFace analysis error: {e}")
                
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
        validated_emotion = "neutral"
    
    finally:
        if cap is not None:
            try:
                cap.release()
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
    
    logger.info("Enhanced emotion detection thread ended")

@csrf_exempt
def update_answer_text(request):
    """Update current answer text for real-time emotion validation"""
    global current_answer_text
    
    if request.method == "POST":
        text = request.POST.get('text', '')
        current_answer_text = text
        
        # Get current validation if validator is available
        validator = get_emotion_validator()
        if validator and text and len(text.strip()) > 10:
            validation = validator.validate_emotion(detected_emotion, text)
            global validated_emotion, emotion_confidence
            validated_emotion = validation.validated_emotion
            emotion_confidence = validation.confidence_score
            
            return JsonResponse({
                'success': True,
                'validated_emotion': validated_emotion,
                'confidence': emotion_confidence,
                'flags': validation.flags,
                'interpretation': validation.contextual_interpretation
            })
        
        return JsonResponse({'success': True, 'message': 'Text updated'})
    
    return JsonResponse({'error': 'Invalid request method'})

# ADD THIS NEW FUNCTION
@csrf_exempt
def get_emotion_status(request):
    """Get current emotion status with validation info"""
    validator = get_emotion_validator()
    
    response_data = {
        'detected_emotion': detected_emotion,
        'validated_emotion': validated_emotion,
        'confidence': emotion_confidence,
        'validator_available': validator is not None
    }
    
    if validator:
        stats = validator.get_validation_stats()
        response_data['validation_stats'] = stats
    
    return JsonResponse(response_data)

# ADD THIS NEW FUNCTION
@csrf_exempt
def correct_emotion(request):
    """Allow users to correct emotion detection for learning"""
    if request.method == "POST":
        original_emotion = request.POST.get('original_emotion', detected_emotion)
        corrected_emotion = request.POST.get('corrected_emotion')
        
        if corrected_emotion:
            validator = get_emotion_validator()
            if validator:
                validator.update_user_feedback(original_emotion, corrected_emotion)
                logger.info(f"User corrected emotion: {original_emotion} -> {corrected_emotion}")
            
            # Update current emotion
            global validated_emotion
            validated_emotion = corrected_emotion
            
            return JsonResponse({
                'success': True,
                'message': 'Emotion correction recorded',
                'new_emotion': corrected_emotion
            })
        
        return JsonResponse({'error': 'Missing corrected emotion'})
    
    return JsonResponse({'error': 'Invalid request method'})

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
                transcript = recognizer.recognize_google(audio_data, language='fil-PH')
                
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

# CONTENT ANALYSIS FUNCTIONS (Enhanced with vectorization)
def analyze_answer_content(answer, question):
    """Deep content analysis of the answer - enhanced for vectorizer integration"""
    analysis = {
        'key_phrases': [],
        'specific_examples': [],
        'action_words': [],
        'outcome_words': [],
        'context_provided': False,
        'quantifiable_results': [],
        'skills_mentioned': [],
        'challenges_discussed': [],
        'learning_mentioned': False
    }
    
    answer_lower = answer.lower()
    
    # Split into sentences
    sentences = answer.split('.')
    
    # Extract specific examples and stories
    story_indicators = [
        r'when i (?:was|worked|had to|needed to|decided to)',
        r'there was a time (?:when|that)',
        r'in my (?:previous|last|current) (?:job|role|position)',
        r'at (?:my|a) (?:company|workplace|job)',
        r'i remember (?:when|a time)',
        r'for example',
        r'specifically'
    ]
    
    for pattern in story_indicators:
        matches = re.findall(pattern, answer_lower)
        if matches:
            analysis['context_provided'] = True
            break
    
    # Extract quantifiable results
    number_patterns = [
        r'\d+%',  # percentages
        r'\$\d+',  # money
        r'\d+ (?:people|team members|clients|users|projects)',
        r'(?:increased|decreased|improved|reduced) by \d+',
        r'\d+ (?:days|weeks|months|years)',
        r'\d+ (?:hours|minutes)'
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, answer_lower)
        analysis['quantifiable_results'].extend(matches)
    
    # Extract action words (what they actually did)
    action_patterns = [
        r'i (?:created|built|designed|implemented|organized|led|managed|developed|solved|analyzed|negotiated|presented|collaborated|coordinated|researched|optimized)',
        r'i (?:decided|chose|recommended|proposed|initiated|established|streamlined|restructured|facilitated|mentored)'
    ]
    
    for pattern in action_patterns:
        matches = re.findall(pattern, answer_lower)
        analysis['action_words'].extend([match.replace('i ', '') for match in matches])
    
    # Extract outcome/result words
    outcome_patterns = [
        r'(?:as a )?result',
        r'(?:this|it) (?:led to|resulted in|caused|helped|improved)',
        r'(?:we|i) (?:achieved|accomplished|delivered|completed|succeeded)',
        r'(?:the outcome|the result) was',
        r'(?:ultimately|finally|in the end)'
    ]
    
    for pattern in outcome_patterns:
        if re.search(pattern, answer_lower):
            analysis['outcome_words'].append(pattern)
    
    # Check for learning/growth mentions
    learning_indicators = ['learned', 'discovered', 'realized', 'understood', 'gained insight', 'now i know', 'next time', 'in future']
    analysis['learning_mentioned'] = any(indicator in answer_lower for indicator in learning_indicators)
    
    # Extract potential skills/technologies mentioned
    common_skills = [
        'leadership', 'communication', 'problem-solving', 'teamwork', 'project management',
        'python', 'javascript', 'react', 'sql', 'aws', 'docker', 'git',
        'agile', 'scrum', 'data analysis', 'machine learning', 'apis'
    ]
    
    for skill in common_skills:
        if skill in answer_lower:
            analysis['skills_mentioned'].append(skill)
    
    # Identify challenges/problems discussed
    challenge_indicators = [
        'challenge', 'problem', 'issue', 'difficulty', 'obstacle', 'setback',
        'conflict', 'disagreement', 'tight deadline', 'limited resources'
    ]
    
    for indicator in challenge_indicators:
        if indicator in answer_lower:
            analysis['challenges_discussed'].append(indicator)
    
    return analysis

def enhanced_analyze_answer_content(answer, question, category="behavioral", role_level="mid"):
    """
    Enhanced version combining existing analysis with semantic vectorization
    """
    
    # Get existing analysis first
    existing_analysis = analyze_answer_content(answer, question)
    
    # Try to get semantic analysis
    vectorizer = get_vectorizer()
    if vectorizer:
        try:
            semantic_score = vectorizer.analyze_answer_semantics(answer, category, role_level)
            
            # Generate contextualized feedback
            semantic_feedback = vectorizer.generate_contextualized_feedback(
                semantic_score, category, role_level, word_limit=180
            )
            
            # Enhance existing analysis with semantic data
            enhanced_analysis = {
                **existing_analysis,  # Keep all existing analysis
                'semantic_analysis': {
                    'content_coverage': semantic_score.content_coverage,
                    'depth_score': semantic_score.depth_score,
                    'relevance_score': semantic_score.relevance_score,
                    'leadership_indicators': semantic_score.leadership_indicators,
                    'problem_solving_indicators': semantic_score.problem_solving_indicators,
                    'technical_depth': semantic_score.technical_depth,
                    'communication_clarity': semantic_score.communication_clarity,
                    'semantic_gaps': semantic_score.gaps,
                    'semantic_strengths': semantic_score.strengths,
                    'semantic_feedback': semantic_feedback
                },
                'overall_score': calculate_overall_score(semantic_score, existing_analysis),
                'enhanced_feedback': generate_combined_feedback(existing_analysis, semantic_score, category)
            }
            
            logger.info(f"Enhanced analysis completed - Overall score: {enhanced_analysis['overall_score']:.2f}")
            return enhanced_analysis
            
        except Exception as e:
            logger.warning(f"Semantic analysis failed, using basic analysis: {e}")
    
    # Fallback to existing analysis if vectorizer fails
    logger.info("Using basic analysis only")
    return {
        **existing_analysis,
        'semantic_analysis': None,
        'overall_score': 0.6,  # Default score
        'enhanced_feedback': "Basic analysis completed successfully."
    }

def calculate_overall_score(semantic_score: SemanticScore, existing_analysis: dict) -> float:
    """Calculate a comprehensive score combining semantic and traditional analysis"""
    
    # Semantic components (60% weight)
    semantic_weight = 0.6
    semantic_avg = (
        semantic_score.content_coverage +
        semantic_score.depth_score +
        semantic_score.relevance_score +
        semantic_score.communication_clarity
    ) / 4
    
    # Traditional analysis components (40% weight)
    traditional_weight = 0.4
    traditional_score = 0.6  # Default
    
    # Extract scores from existing analysis if available
    if 'quantifiable_results' in existing_analysis and existing_analysis['quantifiable_results']:
        traditional_score += 0.1
    if 'action_words' in existing_analysis and len(existing_analysis['action_words']) >= 2:
        traditional_score += 0.1
    if 'context_provided' in existing_analysis and existing_analysis['context_provided']:
        traditional_score += 0.1
    if 'learning_mentioned' in existing_analysis and existing_analysis['learning_mentioned']:
        traditional_score += 0.1
    
    traditional_score = min(traditional_score, 1.0)
    
    # Combine scores
    overall = (semantic_avg * semantic_weight) + (traditional_score * traditional_weight)
    return min(overall, 1.0)

def generate_combined_feedback(existing_analysis: dict, semantic_score: SemanticScore, category: str) -> str:
    """Generate comprehensive feedback combining traditional and semantic analysis"""
    
    feedback_parts = []
    
    # Content quality assessment
    if semantic_score.content_coverage >= 0.7:
        feedback_parts.append("📊 Strong content alignment with interview expectations.")
    elif semantic_score.content_coverage >= 0.5:
        feedback_parts.append("📊 Good content foundation with room for enhancement.")
    else:
        feedback_parts.append("📊 Content needs significant strengthening.")
    
    # Combine semantic strengths with traditional analysis
    combined_strengths = []
    if semantic_score.leadership_indicators >= 0.6:
        combined_strengths.append("leadership presence")
    if semantic_score.problem_solving_indicators >= 0.6:
        combined_strengths.append("problem-solving approach")
    if semantic_score.technical_depth >= 0.6:
        combined_strengths.append("technical depth")
    if existing_analysis.get('quantifiable_results'):
        combined_strengths.append("quantified impact")
    
    if combined_strengths:
        feedback_parts.append(f"✅ Strengths: {', '.join(combined_strengths[:3])}.")
    
    # Combined improvement areas
    improvements = []
    if semantic_score.content_coverage < 0.5:
        improvements.append("strengthen core content themes")
    if semantic_score.depth_score < 0.5:
        improvements.append("add more specific details")
    if not existing_analysis.get('context_provided'):
        improvements.append("provide clearer situation context")
    if semantic_score.communication_clarity < 0.6:
        improvements.append("improve answer structure")
    
    if improvements:
        feedback_parts.append(f"🎯 Focus areas: {', '.join(improvements[:3])}.")
    
    # Category-specific advice
    if category == 'behavioral' and semantic_score.leadership_indicators < 0.4:
        feedback_parts.append("💼 Behavioral tip: Emphasize your personal actions and leadership role.")
    elif category == 'technical' and semantic_score.technical_depth < 0.5:
        feedback_parts.append("🔧 Technical tip: Include specific technologies and implementation details.")
    
    return " ".join(feedback_parts)

def get_detailed_emotion_feedback(emotion, answer):
    """More nuanced emotion feedback based on content"""
    word_count = len(answer.split())
    
    base_feedback = {
        'neutral': "Professional demeanor maintained throughout your response.",
        'happy': "Your positive energy came through well! This enthusiasm is great for interviews.",
        'sad': "You seemed a bit subdued. Try to project more energy and confidence in your delivery.",
        'angry': "You appeared tense. Take deep breaths and focus on staying calm and composed.",
        'fear': "Some nervousness detected - this is normal! Practice will help build confidence.",
        'surprise': "You seemed caught off-guard. Take a moment to collect your thoughts before answering.",
        'disgust': "You appeared uncomfortable. Try to maintain a more positive, engaged expression."
    }
    
    feedback = base_feedback.get(emotion, "Maintain professional composure")
    
    # Add specific delivery advice based on content length and emotion
    if word_count < 30 and emotion in ['fear', 'sad']:
        feedback += " Your brief response combined with this emotion suggests you might need more preparation. Practice your stories out loud."
    elif word_count > 150 and emotion == 'neutral':
        feedback += " Good detail level with calm delivery - this shows confidence in your experience."
    elif emotion == 'happy' and 'challenge' in answer.lower():
        feedback += " Excellent - your positive attitude when discussing challenges shows resilience."
    
    return feedback

# MAIN FEEDBACK FUNCTIONS
def get_vectorized_feedback_with_fallback(user_answer, question, emotion, category="behavioral", role_level="mid"):
    """
    New main feedback function using vectorization when available, falls back gracefully
    """
    
    start_time = time.time()
    
    # Try vectorized analysis first
    vectorizer = get_vectorizer()
    if vectorizer:
        try:
            # Get enhanced analysis
            enhanced_analysis = enhanced_analyze_answer_content(
                user_answer, question, category, role_level
            )
            
            # Generate comprehensive feedback
            feedback_parts = []
            
            # Overall assessment
            overall_score = enhanced_analysis.get('overall_score', 0.6)
            if overall_score >= 0.8:
                feedback_parts.append("🌟 Excellent Response Quality")
            elif overall_score >= 0.6:
                feedback_parts.append("👍 Solid Response Foundation")
            else:
                feedback_parts.append("📈 Response Needs Enhancement")
            
            # Add enhanced feedback
            if enhanced_analysis.get('enhanced_feedback'):
                feedback_parts.append(enhanced_analysis['enhanced_feedback'])
            
            # Add semantic insights if available
            semantic_analysis = enhanced_analysis.get('semantic_analysis')
            if semantic_analysis and semantic_analysis.get('semantic_feedback'):
                feedback_parts.append(f"🧠 Semantic Analysis: {semantic_analysis['semantic_feedback']}")
            
            # Add emotion feedback
            emotion_feedback = get_detailed_emotion_feedback(emotion, user_answer)
            feedback_parts.append(f"🎭 Delivery: {emotion_feedback}")
            
            # Performance metrics
            processing_time = time.time() - start_time
            logger.info(f"Vectorized feedback completed in {processing_time:.2f}s")
            
            return "\n\n".join(feedback_parts)
            
        except Exception as e:
            logger.warning(f"Vectorized feedback failed: {e}, falling back to basic analysis")
    
    # Fallback to basic feedback
    logger.info("Using basic feedback system")
    return get_basic_feedback(user_answer, question, emotion, category)

def get_basic_feedback(user_answer, question, emotion, category):
    """Basic feedback when vectorizer is not available"""
    
    # Get basic content analysis
    analysis = analyze_answer_content(user_answer, question)
    
    feedback_parts = []
    
    # Basic length assessment
    word_count = len(user_answer.split())
    if word_count < 50:
        feedback_parts.append("📏 Your answer is quite brief. Consider adding more specific details and examples.")
    elif word_count > 200:
        feedback_parts.append("📏 Good detail level! Stay focused on key points.")
    else:
        feedback_parts.append("📏 Good answer length - detailed but concise.")
    
    # Content strengths
    strengths = []
    if analysis['context_provided']:
        strengths.append("good context")
    if analysis['quantifiable_results']:
        strengths.append("specific metrics")
    if analysis['action_words']:
        strengths.append("clear actions")
    if analysis['learning_mentioned']:
        strengths.append("learning mindset")
    
    if strengths:
        feedback_parts.append(f"✅ Strengths: {', '.join(strengths)}.")
    
    # Areas for improvement
    improvements = []
    if not analysis['context_provided']:
        improvements.append("add more situation context")
    if not analysis['quantifiable_results']:
        improvements.append("include measurable outcomes")
    if len(analysis['action_words']) < 2:
        improvements.append("use more specific action words")
    
    if improvements:
        feedback_parts.append(f"🎯 Improvements: {', '.join(improvements)}.")
    
    # Emotion feedback
    emotion_feedback = get_detailed_emotion_feedback(emotion, user_answer)
    feedback_parts.append(f"🎭 Delivery: {emotion_feedback}")
    
    return "\n\n".join(feedback_parts)

# MAIN VIEW FUNCTIONS
# UPDATE THIS FUNCTION - Replace your existing feedback_page function
def feedback_page(request):
    """Enhanced feedback page with emotion validation"""
    global stop_event, emotion_thread, validated_emotion, emotion_confidence
    
    # Stop emotion detection
    stop_event.set()
    if emotion_thread and emotion_thread.is_alive():
        emotion_thread.join(timeout=5)
    
    question = request.session.get('question', '')
    answer = request.GET.get('transcript') or request.session.get('answer', '')
    request.session['answer'] = answer
    
    # Use validated emotion instead of raw emotion
    final_emotion = validated_emotion
    category = request.session.get('category', 'behavioral')
    role_level = request.session.get('role_level', 'mid')
    
    logger.info(f"Generating feedback with validated emotion: {final_emotion} (confidence: {emotion_confidence:.3f})")
    
    start_time = time.time()
    
    # Get emotion validation details
    validator = get_emotion_validator()
    validation_details = None
    if validator and answer:
        validation = validator.validate_emotion(detected_emotion, answer)
        validation_details = {
            'original_emotion': validation.original_emotion,
            'validated_emotion': validation.validated_emotion,
            'confidence_score': validation.confidence_score,
            'sentiment_score': validation.sentiment_score,
            'correlation_strength': validation.correlation_strength,
            'contextual_interpretation': validation.contextual_interpretation,
            'flags': validation.flags,
            'adjustment_reason': validation.adjustment_reason
        }
    
    # Use the enhanced feedback system with validated emotion
    feedback = get_vectorized_feedback_with_fallback(
        answer, question, final_emotion, category, role_level
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get additional metrics
    vectorizer = get_vectorizer()
    metrics = {}
    if vectorizer:
        try:
            semantic_score = vectorizer.analyze_answer_semantics(answer, category, role_level)
            metrics = {
                'content_coverage': f"{semantic_score.content_coverage:.1%}",
                'technical_depth': f"{semantic_score.technical_depth:.1%}",
                'leadership_indicators': f"{semantic_score.leadership_indicators:.1%}",
                'communication_clarity': f"{semantic_score.communication_clarity:.1%}",
                'emotion_confidence': f"{emotion_confidence:.1%}"
            }
        except Exception as e:
            logger.warning(f"Could not generate metrics: {e}")
    
    return render(request, 'feedback.html', {
        'question': question,
        'answer': answer,
        'emotion': final_emotion,
        'raw_emotion': detected_emotion,
        'emotion_confidence': emotion_confidence,
        'feedback': feedback,
        'feedback_type': "Enhanced with Emotion Validation" if validator else "Enhanced Analysis",
        'category': category,
        'role_level': role_level,
        'camera_available': camera_available,
        'processing_time': f"{processing_time:.1f}s",
        'metrics': metrics,
        'vectorizer_available': vectorizer is not None,
        'validator_available': validator is not None,
        'validation_details': validation_details
    })

# UPDATE THIS FUNCTION - Enhance your existing get_detailed_emotion_feedback
def get_detailed_emotion_feedback(emotion, answer, confidence=1.0, flags=None):
    """Enhanced emotion feedback with validation context"""
    word_count = len(answer.split())
    flags = flags or []
    
    base_feedback = {
        'neutral': "Professional demeanor maintained throughout your response.",
        'happy': "Your positive energy came through well! This enthusiasm is great for interviews.",
        'sad': "You seemed a bit subdued. Try to project more energy and confidence in your delivery.",
        'angry': "You appeared tense. Take deep breaths and focus on staying calm and composed.",
        'fear': "Some nervousness detected - this is normal! Practice will help build confidence.",
        'surprise': "You seemed caught off-guard. Take a moment to collect your thoughts before answering.",
        'disgust': "You appeared uncomfortable. Try to maintain a more positive, engaged expression."
    }
    
    feedback = base_feedback.get(emotion, "Maintain professional composure")
    
    # Add confidence context
    if confidence < 0.4:
        feedback += f" (Note: Low confidence detection - {confidence:.1%})"
    elif confidence >= 0.8:
        feedback += f" (High confidence detection - {confidence:.1%})"
    
    # Add flag-specific feedback
    if flags:
        flag_feedback = []
        for flag in flags[:2]:  # Show max 2 flags
            if "Technical discussion" in flag:
                flag_feedback.append("Your focused technical discussion shows expertise.")
            elif "Problem-solving" in flag:
                flag_feedback.append("Your methodical problem-solving approach is impressive.")
            elif "High engagement" in flag:
                flag_feedback.append("Your passion for the topic comes through clearly.")
            elif "Concentration" in flag:
                flag_feedback.append("Your thoughtful consideration of the question shows depth.")
        
        if flag_feedback:
            feedback += " " + " ".join(flag_feedback)
    
    # Add specific delivery advice
    if word_count < 30 and emotion in ['fear', 'sad'] and confidence > 0.6:
        feedback += " Your brief response suggests you might need more preparation. Practice your stories out loud."
    elif word_count > 150 and emotion == 'neutral' and confidence > 0.7:
        feedback += " Good detail level with calm delivery - this shows confidence in your experience."
    elif emotion == 'happy' and 'challenge' in answer.lower() and confidence > 0.6:
        feedback += " Excellent - your positive attitude when discussing challenges shows resilience."
    
    return feedback
# UTILITY FUNCTIONS
@csrf_exempt
def set_role_level(request):
    """Endpoint to set target role level"""
    if request.method == "POST":
        role_level = request.POST.get('role_level', 'mid')
        if role_level in ['junior', 'mid', 'senior', 'lead']:
            request.session['role_level'] = role_level
            return JsonResponse({'success': True, 'role_level': role_level})
        else:
            return JsonResponse({'error': 'Invalid role level'})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def vectorization_status(request):
    """Check vectorization system status"""
    vectorizer = get_vectorizer()
    
    dependencies = {
        'sentence_transformers': False,
        'sklearn': False,
        'numpy': False
    }
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    return JsonResponse({
        'vectorizer_available': vectorizer is not None,
        'dependencies': dependencies,
        'fallback_mode': vectorizer.fallback_mode if vectorizer else True,
        'cache_dir': vectorizer.cache_dir if vectorizer else None,
        'model_loaded': bool(vectorizer and vectorizer.model) if vectorizer else False
    })

# OLLAMA INTEGRATION (Optional - kept for backward compatibility)
def get_ollama_feedback(user_answer, question, emotion, category="behavioral"):
    """Optional Ollama integration - kept for users who want AI feedback"""
    logger.info("Attempting Ollama feedback...")
    
    try:
        # Check if Ollama is available
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            return None
        
        # Simple prompt for faster processing
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "mistral",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an interview coach. Evaluate {category} interview answers. Be encouraging but specific."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {user_answer}\nEmotion: {emotion}\n\nProvide brief feedback on strengths, improvements, and delivery."
                }
            ]
        }
        
        response = requests.post(url, json=payload, timeout=25)
        if response.status_code == 200:
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                return result['message']['content']
    
    except Exception as e:
        logger.warning(f"Ollama feedback failed: {e}")
    
    return None

def get_enhanced_feedback_with_ai_fallback(user_answer, question, emotion, category="behavioral", role_level="mid"):
    """
    Enhanced feedback that tries vectorization first, then Ollama, then basic analysis
    """
    
    # Try vectorized feedback first
    vectorizer = get_vectorizer()
    if vectorizer:
        try:
            logger.info("Using vectorized feedback system")
            return get_vectorized_feedback_with_fallback(user_answer, question, emotion, category, role_level)
        except Exception as e:
            logger.warning(f"Vectorized feedback failed: {e}")
    
    # Try Ollama as fallback
    ollama_feedback = get_ollama_feedback(user_answer, question, emotion, category)
    if ollama_feedback:
        logger.info("Using Ollama AI feedback")
        emotion_feedback = get_detailed_emotion_feedback(emotion, user_answer)
        return f"🤖 AI Analysis:\n\n{ollama_feedback}\n\n🎭 Delivery: {emotion_feedback}"
    
    # Final fallback to basic analysis
    logger.info("Using basic feedback system")
    return get_basic_feedback(user_answer, question, emotion, category)

# OPTIONAL: Enhanced interview page with role level selection
def interview_page_with_role_selection(request):
    """Enhanced interview page that includes role level selection"""
    global stop_event, emotion_thread, emotion_detection_enabled
    
    category = request.GET.get('category', 'behavioral')
    role_level = request.GET.get('role_level', 'mid')
    
    # Store role level in session
    request.session['role_level'] = role_level
    
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
        'category': category,
        'role_level': role_level,
        'camera_available': camera_available,
        'emotion_detection_enabled': emotion_detection_enabled,
        'vectorizer_available': get_vectorizer() is not None
    })

# DIAGNOSTIC FUNCTIONS
@csrf_exempt
def diagnostic_info(request):
    """Endpoint to get diagnostic information about the system"""
    vectorizer = get_vectorizer()
    
    info = {
        'system': {
            'platform': platform.system(),
            'camera_available': camera_available,
            'emotion_detection_enabled': emotion_detection_enabled
        },
        'vectorizer': {
            'available': vectorizer is not None,
            'fallback_mode': vectorizer.fallback_mode if vectorizer else True,
            'model_loaded': bool(vectorizer and vectorizer.model) if vectorizer else False,
            'cache_dir': vectorizer.cache_dir if vectorizer else None
        },
        'dependencies': {
            'opencv': True,  # We know this works since we imported cv2
            'deepface': True,  # We know this works since we imported DeepFace
            'speech_recognition': True,  # We know this works
            'sentence_transformers': False,
            'sklearn': False,
            'numpy': False
        }
    }
    
    # Check optional dependencies
    try:
        import sentence_transformers
        info['dependencies']['sentence_transformers'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        info['dependencies']['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        info['dependencies']['numpy'] = True
    except ImportError:
        pass
    
    # Check Ollama availability
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        info['ollama'] = {
            'available': response.status_code == 200,
            'models': response.json().get('models', []) if response.status_code == 200 else []
        }
    except Exception:
        info['ollama'] = {'available': False, 'models': []}
    
    return JsonResponse(info)

# INITIALIZATION HELPER
def initialize_vectorizer_on_startup():
    """Call this function when Django starts up to initialize the vectorizer"""
    try:
        vectorizer = get_vectorizer()
        if vectorizer:
            logger.info("Vectorizer initialized successfully on startup")
            return True
        else:
            logger.warning("Vectorizer initialization failed on startup")
            return False
    except Exception as e:
        logger.error(f"Error during vectorizer startup initialization: {e}")
        return False

# USAGE INSTRUCTIONS
"""
INTEGRATION INSTRUCTIONS:

1. REMOVED FROM YOUR ORIGINAL VIEWS.PY:
   - Redundant feedback functions (get_fallback_feedback, get_enhanced_personalized_feedback, etc.)
   - Duplicate analysis functions that are now handled by the vectorizer
   - Old OLLAMA functions that were causing timeouts
   - Verbose feedback generation that's now streamlined

2. NEW MAIN FUNCTIONS TO USE:
   - feedback_page(): Main feedback function (replaces your old one)
   - get_vectorized_feedback_with_fallback(): Main feedback generation
   - enhanced_analyze_answer_content(): Enhanced content analysis

3. URL PATTERNS TO ADD/UPDATE IN urls.py:
   urlpatterns = [
       path('', start_page, name='start'),
       path('interview/', interview_page, name='interview'),
       path('feedback/', feedback_page, name='feedback'),
       path('transcribe/', transcribe_audio, name='transcribe'),
       path('set-emotion/', set_emotion, name='set_emotion'),
       path('set-role-level/', set_role_level, name='set_role_level'),
       path('vectorization-status/', vectorization_status, name='vectorization_status'),
       path('diagnostic/', diagnostic_info, name='diagnostic'),
   ]

4. TEMPLATE UPDATES NEEDED:
   - Add role_level selection to your interview.html
   - Update feedback.html to show metrics if available
   - Add vectorizer status indicators

5. DEPENDENCIES TO INSTALL (optional but recommended):
   pip install sentence-transformers scikit-learn numpy

6. TO INITIALIZE ON DJANGO STARTUP:
   Add to your apps.py ready() method:
   from .views import initialize_vectorizer_on_startup
   initialize_vectorizer_on_startup()
"""