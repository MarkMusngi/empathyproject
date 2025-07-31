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
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from .models import UserProfile, InterviewSession
from .feedback_models import FeedbackModelManager, create_feedback_context
from django.db import transaction

from django.db import models
# Import the content vectorizer
from .content_vectorizer import ContentVectorizer, SemanticScore


logger = logging.getLogger(__name__)

# Global vectorizer instance - initialize once when Django starts
_vectorizer_instance = None
validated_emotion = "neutral"
emotion_confidence = 1.0
current_answer_text = ""
emotion_session_history = []
user_corrections = {}
_feedback_manager = None
def make_json_safe(value):
    """Convert numpy types and other non-serializable types to JSON-safe types"""
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif hasattr(value, 'item'):  # NumPy scalar
        return value.item()
    elif isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_safe(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(make_json_safe(item) for item in value)
    return value
def get_feedback_manager():
    """Singleton pattern for feedback model manager"""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackModelManager()
    return _feedback_manager

@dataclass
class EmotionReading:
    """Single emotion detection reading with metadata"""
    emotion: str
    confidence: float
    timestamp: float
    raw_scores: Dict[str, float]  # All emotion scores from DeepFace

@dataclass
class EmotionMetrics:
    """Emotion stability and confidence metrics"""
    current_emotion: str
    smoothed_confidence: float
    stability_score: float  # How stable the emotion has been
    transition_detected: bool
    time_in_current_emotion: float
    recent_transitions: int
    processing_performance: Dict[str, float]

class EmotionSmoother:
    """Handles temporal smoothing and confidence scoring for emotion detection"""
    
    def __init__(self, window_size=10, min_confidence=0.3, transition_threshold=0.6):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.transition_threshold = transition_threshold
        
        # Sliding window for emotion readings
        self.emotion_history = deque(maxlen=window_size)
        
        # Smoothing parameters
        self.recent_bias = 0.3  # How much to weight recent readings
        self.stability_window = 5  # Frames to consider for stability
        
        # Performance tracking
        self.processing_times = deque(maxlen=20)
        self.frame_skip_count = 0
        self.total_frames = 0
        
        # Current state
        self.current_stable_emotion = "neutral"
        self.emotion_start_time = time.time()
        self.last_transition_time = 0
        self.transition_count = 0
        
        # Adaptive sampling
        self.base_sample_rate = 0.2  # 200ms between samples
        self.current_sample_rate = self.base_sample_rate
        self.max_sample_rate = 1.0   # 1 second max between samples
        self.min_sample_rate = 0.1   # 100ms min between samples
    
    def add_reading(self, emotion: str, raw_scores: Dict[str, float], processing_time: float) -> EmotionMetrics:
        """Add new emotion reading and return smoothed metrics"""
        current_time = time.time()
        
        # Calculate confidence from raw scores
        confidence = self._calculate_confidence(emotion, raw_scores)
        
        # Create reading
        reading = EmotionReading(
            emotion=emotion,
            confidence=confidence,
            timestamp=current_time,
            raw_scores=raw_scores.copy()
        )
        
        # Add to history
        self.emotion_history.append(reading)
        self.processing_times.append(processing_time)
        self.total_frames += 1
        
        # Calculate smoothed emotion
        smoothed_emotion, smoothed_confidence = self._calculate_smoothed_emotion()
        
        # Detect transitions
        transition_detected = self._detect_transition(smoothed_emotion)
        
        # Calculate stability metrics
        stability_score = self._calculate_stability()
        time_in_emotion = current_time - self.emotion_start_time
        recent_transitions = self._count_recent_transitions()
        
        # Update adaptive sampling rate
        self._update_sampling_rate(stability_score, smoothed_confidence)
        
        # Create metrics
        metrics = EmotionMetrics(
            current_emotion=smoothed_emotion,
            smoothed_confidence=smoothed_confidence,
            stability_score=stability_score,
            transition_detected=transition_detected,
            time_in_current_emotion=time_in_emotion,
            recent_transitions=recent_transitions,
            processing_performance={
                'avg_processing_time': np.mean(list(self.processing_times)) if self.processing_times else 0,
                'current_sample_rate': self.current_sample_rate,
                'frame_skip_rate': self.frame_skip_count / max(self.total_frames, 1),
                'stability_score': stability_score
            }
        )
        
        return metrics
    
    def _calculate_confidence(self, emotion: str, raw_scores: Dict[str, float]) -> float:
        """Calculate confidence score from raw emotion scores"""
        if not raw_scores:
            return 0.5
        
        # Get the score for the detected emotion
        dominant_score = raw_scores.get(emotion, 0) / 100.0  # DeepFace returns 0-100
        
        # Calculate score separation (how much higher than second-best)
        sorted_scores = sorted(raw_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            separation = (sorted_scores[0] - sorted_scores[1]) / 100.0
            confidence = (dominant_score + separation) / 2.0
        else:
            confidence = dominant_score
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_smoothed_emotion(self) -> Tuple[str, float]:
        """Calculate smoothed emotion using weighted averaging"""
        if not self.emotion_history:
            return "neutral", 0.5
        
        if len(self.emotion_history) == 1:
            reading = self.emotion_history[0]
            return reading.emotion, reading.confidence
        
        # Weight recent readings more heavily
        current_time = time.time()
        emotion_scores = {}
        total_weight = 0
        
        for i, reading in enumerate(self.emotion_history):
            # Time-based weight (more recent = higher weight)
            age = current_time - reading.timestamp
            time_weight = np.exp(-age / 2.0)  # Decay over 2 seconds
            
            # Position-based weight (recent bias)
            position_weight = 1.0 + (i / len(self.emotion_history)) * self.recent_bias
            
            # Confidence weight
            confidence_weight = reading.confidence
            
            # Combined weight
            weight = time_weight * position_weight * confidence_weight
            
            # Add to emotion scores
            if reading.emotion not in emotion_scores:
                emotion_scores[reading.emotion] = 0
            emotion_scores[reading.emotion] += weight
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_weight
        
        # Find dominant emotion
        if not emotion_scores:
            return "neutral", 0.5
        
        smoothed_emotion = max(emotion_scores.keys(), key=lambda e: emotion_scores[e])
        smoothed_confidence = emotion_scores[smoothed_emotion]
        
        return smoothed_emotion, smoothed_confidence
    
    def _detect_transition(self, new_emotion: str) -> bool:
        """Detect if there's been a significant emotion transition"""
        current_time = time.time()
        
        # Check if emotion changed from current stable emotion
        if new_emotion != self.current_stable_emotion:
            # Only consider it a transition if we have enough confidence
            recent_readings = list(self.emotion_history)[-self.stability_window:]
            
            # Count how many recent readings support the new emotion
            support_count = sum(1 for r in recent_readings if r.emotion == new_emotion)
            support_ratio = support_count / len(recent_readings) if recent_readings else 0
            
            if support_ratio >= self.transition_threshold:
                # Significant transition detected
                self.last_transition_time = current_time
                self.emotion_start_time = current_time
                self.current_stable_emotion = new_emotion
                self.transition_count += 1
                
                logger.info(f"Emotion transition detected: -> {new_emotion} "
                           f"(support: {support_ratio:.2f})")
                return True
        
        return False
    
    def _calculate_stability(self) -> float:
        """Calculate emotion stability score (0-1, higher = more stable)"""
        if len(self.emotion_history) < 2:
            return 1.0
        
        # Look at recent readings
        recent_readings = list(self.emotion_history)[-self.stability_window:]
        
        # Calculate consistency
        if not recent_readings:
            return 1.0
        
        # Count how many readings match the most common emotion
        emotions = [r.emotion for r in recent_readings]
        most_common = max(set(emotions), key=emotions.count)
        consistency = emotions.count(most_common) / len(emotions)
        
        # Factor in confidence levels
        avg_confidence = np.mean([r.confidence for r in recent_readings])
        
        # Factor in time stability (longer in same emotion = more stable)
        time_stability = min(time.time() - self.emotion_start_time, 10.0) / 10.0
        
        # Combined stability score
        stability = (consistency * 0.5) + (avg_confidence * 0.3) + (time_stability * 0.2)
        
        return min(max(stability, 0.0), 1.0)
    
    def _count_recent_transitions(self, window_seconds=30) -> int:
        """Count transitions in recent time window"""
        current_time = time.time()
        recent_transition_count = 0
        
        # Count transitions in emotion history within window
        if len(self.emotion_history) < 2:
            return 0
        
        prev_emotion = None
        for reading in self.emotion_history:
            if current_time - reading.timestamp > window_seconds:
                continue
            
            if prev_emotion and reading.emotion != prev_emotion:
                recent_transition_count += 1
            prev_emotion = reading.emotion
        
        return recent_transition_count
    
    def _update_sampling_rate(self, stability_score: float, confidence: float):
        """Adaptively adjust sampling rate based on stability and confidence"""
        # High stability + high confidence = can sample less frequently
        # Low stability + low confidence = need more frequent sampling
        
        stability_factor = stability_score
        confidence_factor = confidence
        
        # Combined factor (higher = more stable/confident = can sample slower)
        combined_factor = (stability_factor + confidence_factor) / 2.0
        
        # Adaptive rate: unstable/low confidence = faster sampling
        if combined_factor < 0.3:
            # Very unstable - sample frequently
            self.current_sample_rate = self.min_sample_rate
        elif combined_factor < 0.6:
            # Moderately stable - normal sampling
            self.current_sample_rate = self.base_sample_rate
        else:
            # Very stable - can sample less frequently
            self.current_sample_rate = min(
                self.base_sample_rate * (1 + combined_factor),
                self.max_sample_rate
            )
    
    def should_skip_frame(self, last_sample_time: float) -> bool:
        """Determine if current frame should be skipped based on adaptive sampling"""
        current_time = time.time()
        time_since_last = current_time - last_sample_time
        
        should_skip = time_since_last < self.current_sample_rate
        
        if should_skip:
            self.frame_skip_count += 1
        
        return should_skip
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for monitoring"""
        if not self.processing_times:
            return {'status': 'no_data'}
        
        avg_processing = np.mean(list(self.processing_times))
        
        return {
            'avg_processing_time_ms': avg_processing * 1000,
            'current_sample_rate_ms': self.current_sample_rate * 1000,
            'frame_skip_percentage': (self.frame_skip_count / max(self.total_frames, 1)) * 100,
            'total_frames_processed': self.total_frames,
            'emotion_transitions': self.transition_count,
            'current_stability': self._calculate_stability(),
            'status': 'optimal' if avg_processing < 0.5 else 'acceptable' if avg_processing < 1.0 else 'slow'
        }


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
    
_emotion_smoother = EmotionSmoother()

def detect_emotion_live():
    """UPDATED: Enhanced emotion detection with temporal smoothing and confidence scoring"""
    global detected_emotion, validated_emotion, emotion_confidence, camera_available, current_answer_text
    global _emotion_smoother
    
    working_backend = check_camera_availability()
    if working_backend is None:
        logger.warning("Skipping emotion detection - no working camera backend")
        detected_emotion = "neutral"
        validated_emotion = "neutral"
        return
    
    cap = None
    validator = get_emotion_validator()
    last_sample_time = 0
    last_validation_time = 0
    
    try:
        cap = cv2.VideoCapture(0, working_backend)
        if not cap.isOpened():
            logger.warning("Could not open camera for emotion detection")
            detected_emotion = "neutral"
            validated_emotion = "neutral"
            return
        
        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Higher FPS for better temporal resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        consecutive_failures = 0
        max_failures = 5
        
        logger.info("Starting enhanced emotion detection with temporal smoothing")
        
        while not stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Frame read failed (attempt {consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.warning("Too many camera read failures, stopping emotion detection")
                        break
                    time.sleep(0.2)
                    continue
                
                consecutive_failures = 0
                
                # Check if we should skip this frame (adaptive sampling)
                if _emotion_smoother.should_skip_frame(last_sample_time):
                    time.sleep(0.05)  # Short sleep to prevent busy waiting
                    continue
                
                last_sample_time = frame_start_time
                
                # Analyze emotion with timing
                analysis_start = time.time()
                
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    
                    if result and len(result) > 0:
                        analysis_time = time.time() - analysis_start
                        
                        raw_emotion = result[0].get('dominant_emotion', 'neutral')
                        raw_scores = result[0].get('emotion', {})
                        
                        # Add reading to smoother
                        metrics = _emotion_smoother.add_reading(raw_emotion, raw_scores, analysis_time)
                        
                        # Update global emotion state with smoothed values
                        smoothed_emotion = metrics.current_emotion
                        smoothed_confidence = metrics.smoothed_confidence
                        
                        # Only update if significantly different or more confident
                        if (smoothed_emotion != detected_emotion or 
                            abs(smoothed_confidence - emotion_confidence) > 0.1):
                            
                            detected_emotion = smoothed_emotion
                            
                            # Log significant changes
                            if metrics.transition_detected:
                                logger.info(f"Smoothed emotion transition: {smoothed_emotion} "
                                           f"(confidence: {smoothed_confidence:.3f}, "
                                           f"stability: {metrics.stability_score:.3f})")
                        
                        # Validate emotion periodically with text context
                        current_time = time.time()
                        if (validator and current_answer_text and 
                            current_time - last_validation_time > 2.0):  # Every 2 seconds
                            
                            validation = validator.validate_emotion(
                                smoothed_emotion, current_answer_text
                            )
                            
                            validated_emotion = validation.validated_emotion
                            
                            # Combine smoother confidence with validation confidence
                            emotion_confidence = (smoothed_confidence + validation.confidence_score) / 2.0
                            
                            if validation.flags:
                                logger.debug(f"Validation flags: {validation.flags}")
                            
                            last_validation_time = current_time
                        else:
                            # Use smoothed values without validation
                            validated_emotion = smoothed_emotion
                            emotion_confidence = smoothed_confidence
                        
                        # Log performance occasionally
                        if _emotion_smoother.total_frames % 50 == 0:
                            perf = _emotion_smoother.get_performance_summary()
                            logger.info(f"Emotion detection performance: {perf['status']} "
                                       f"({perf['avg_processing_time_ms']:.1f}ms avg, "
                                       f"{perf['frame_skip_percentage']:.1f}% skipped)")
                        
                except Exception as e:
                    logger.warning(f"DeepFace analysis error: {e}")
                    analysis_time = time.time() - analysis_start
                
                # Adaptive sleep based on processing performance
                processing_time = time.time() - frame_start_time
                if processing_time < _emotion_smoother.current_sample_rate:
                    time.sleep(_emotion_smoother.current_sample_rate - processing_time)
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Emotion detection frame error: {e}")
                if consecutive_failures >= max_failures:
                    logger.warning("Too many consecutive failures, stopping emotion detection")
                    break
                time.sleep(0.2)
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
        
        # Log final performance summary
        perf = _emotion_smoother.get_performance_summary()
        logger.info(f"Emotion detection session ended. Final performance: {perf}")
    
    logger.info("Enhanced emotion detection with smoothing ended")


@csrf_exempt
def get_emotion_metrics(request):
    """Get detailed emotion metrics including smoothing statistics - FIXED"""
    global _emotion_smoother
    
    perf = _emotion_smoother.get_performance_summary()
    current_time = time.time()
    
    # Get recent emotion history
    recent_emotions = []
    for reading in list(_emotion_smoother.emotion_history)[-5:]:
        recent_emotions.append({
            'emotion': reading.emotion,
            'confidence': make_json_safe(reading.confidence),
            'timestamp': make_json_safe(reading.timestamp),
            'age_seconds': make_json_safe(current_time - reading.timestamp)
        })

    stability_score = _emotion_smoother._calculate_stability()
    
    response_data = {
        'current_emotion': str(validated_emotion),
        'raw_emotion': str(detected_emotion),
        'confidence': make_json_safe(emotion_confidence),
        'stability_score': make_json_safe(stability_score),
        'time_in_current_emotion': make_json_safe(current_time - _emotion_smoother.emotion_start_time),
        'recent_transitions': make_json_safe(_emotion_smoother._count_recent_transitions()),
        'recent_emotion_history': recent_emotions,
        'performance': make_json_safe(perf),
        'smoothing_active': len(_emotion_smoother.emotion_history) > 1,
        'adaptive_sampling': {
            'current_rate_ms': make_json_safe(_emotion_smoother.current_sample_rate * 1000),
            'base_rate_ms': make_json_safe(_emotion_smoother.base_sample_rate * 1000),
            'skip_rate_percentage': make_json_safe(
                (getattr(_emotion_smoother, 'frame_skip_count', 0) /
                 max(getattr(_emotion_smoother, 'total_frames', 1), 1)) * 100
            )
        }
    }

    return JsonResponse(response_data)
def reset_emotion_smoother():
    """NEW: Reset emotion smoother state (call when starting new interview)"""
    global _emotion_smoother
    _emotion_smoother = EmotionSmoother()
    logger.info("Emotion smoother reset for new session")

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
    """UPDATED: Handle specific question parameter and reset smoother when starting new interview"""
    global stop_event, emotion_thread, emotion_detection_enabled
    
    category = request.GET.get('category', 'behavioral')
    
    # Check if a specific question was passed (for retry functionality)
    specific_question = request.GET.get('question')
    
    if specific_question:
        # Use the specific question passed in the URL
        question = specific_question
    else:
        # Select a random question from the category
        questions = interview_questions.get(category, interview_questions['behavioral'])
        question = random.choice(questions)
    
    # Store in session
    request.session['question'] = question
    request.session['category'] = category

    # Reset emotion smoother for new session
    reset_emotion_smoother()

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
        logger.info("Started enhanced emotion detection thread with smoothing")
    else:
        logger.info("Emotion detection disabled by user")

    return render(request, 'interview.html', {
        'question': question,
        'category': category,
        'camera_available': camera_available,
        'emotion_detection_enabled': emotion_detection_enabled,
        'is_retry': bool(specific_question)  # Flag to indicate if this is a retry
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

@csrf_exempt
def emotion_confirmation_overlay(request):
    """Get current emotion with confidence for real-time overlay"""
    global detected_emotion, validated_emotion, emotion_confidence, _emotion_smoother
    
    current_time = time.time()
    
    # Get recent emotion stability
    stability_score = _emotion_smoother._calculate_stability()
    recent_transitions = _emotion_smoother._count_recent_transitions(window_seconds=10)
    
    # Determine if emotion needs confirmation
    needs_confirmation = (
        emotion_confidence < 0.6 or  # Low confidence
        stability_score < 0.5 or     # Unstable
        recent_transitions > 2       # Too many transitions
    )
    
    response_data = {
        'detected_emotion': detected_emotion,
        'validated_emotion': validated_emotion,
        'confidence': float(emotion_confidence),
        'stability_score': float(stability_score),
        'needs_confirmation': bool(needs_confirmation),
        'recent_transitions': recent_transitions,
        'time_in_current_emotion': float(current_time - _emotion_smoother.emotion_start_time),
        'overlay_message': get_overlay_message(detected_emotion, emotion_confidence, stability_score)
    }
    
    return JsonResponse(response_data)

def get_overlay_message(emotion, confidence, stability):
    """Generate appropriate overlay message based on emotion state"""
    if confidence >= 0.8 and stability >= 0.7:
        return f"✓ {emotion.title()} - High confidence"
    elif confidence >= 0.5 and stability >= 0.5:
        return f"~ {emotion.title()} - Medium confidence"
    elif confidence < 0.4:
        return f"? {emotion.title()} - Low confidence - Please confirm"
    else:
        return f"⚡ {emotion.title()} - Unstable - Please confirm"

@csrf_exempt
def quick_emotion_correction(request):
    """Handle quick emotion corrections during interview"""
    global validated_emotion, emotion_confidence, emotion_session_history
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            original_emotion = data.get('original_emotion', detected_emotion)
            corrected_emotion = data.get('corrected_emotion')
            correction_reason = data.get('reason', 'user_correction')
            timestamp = data.get('timestamp', time.time())
            
            if not corrected_emotion:
                return JsonResponse({'error': 'Missing corrected emotion'})
            
            # Update current emotion
            validated_emotion = corrected_emotion
            emotion_confidence = 0.9  # High confidence for user corrections
            
            # Store correction in session history
            correction_entry = {
                'timestamp': timestamp,
                'original_emotion': original_emotion,
                'corrected_emotion': corrected_emotion,
                'reason': correction_reason,
                'confidence_before': emotion_confidence,
                'user_corrected': True
            }
            emotion_session_history.append(correction_entry)
            
            # Update user correction statistics
            correction_key = f"{original_emotion}->{corrected_emotion}"
            if correction_key not in user_corrections:
                user_corrections[correction_key] = 0
            user_corrections[correction_key] += 1
            
            # Update emotion validator if available
            validator = get_emotion_validator()
            if validator:
                validator.update_user_feedback(original_emotion, corrected_emotion)
            
            logger.info(f"Quick emotion correction: {original_emotion} -> {corrected_emotion}")
            
            return JsonResponse({
                'success': True,
                'new_emotion': corrected_emotion,
                'confidence': emotion_confidence,
                'message': f'Emotion updated to {corrected_emotion}',
                'history_length': len(emotion_session_history)
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'})
        except Exception as e:
            logger.error(f"Quick correction error: {e}")
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def emotion_history_timeline(request):
    """Get emotion history for timeline display"""
    global emotion_session_history, _emotion_smoother
    
    # Get recent emotion readings from smoother
    recent_readings = []
    current_time = time.time()
    
    for reading in list(_emotion_smoother.emotion_history):
        recent_readings.append({
            'timestamp': reading.timestamp,
            'emotion': reading.emotion,
            'confidence': float(reading.confidence),
            'age_seconds': float(current_time - reading.timestamp),
            'user_corrected': False
        })
    
    # Combine with session history (user corrections)
    timeline_data = recent_readings + [
        {
            'timestamp': entry['timestamp'],
            'emotion': entry['corrected_emotion'],
            'confidence': 0.9,
            'age_seconds': float(current_time - entry['timestamp']),
            'user_corrected': True,
            'original_emotion': entry['original_emotion'],
            'correction_reason': entry['reason']
        }
        for entry in emotion_session_history
    ]
    
    # Sort by timestamp and limit to last 50 entries
    timeline_data.sort(key=lambda x: x['timestamp'], reverse=True)
    timeline_data = timeline_data[:50]
    
    # Calculate session statistics
    total_readings = len(timeline_data)
    user_corrections_count = len([x for x in timeline_data if x['user_corrected']])
    accuracy = ((total_readings - user_corrections_count) / max(total_readings, 1)) * 100
    
    return JsonResponse({
        'timeline': timeline_data,
        'statistics': {
            'total_readings': total_readings,
            'user_corrections': user_corrections_count,
            'detection_accuracy': round(accuracy, 1),
            'session_duration': float(current_time - _emotion_smoother.emotion_start_time) if _emotion_smoother.emotion_history else 0
        },
        'most_common_corrections': dict(list(user_corrections.items())[:5])
    })

@csrf_exempt
def emotion_impact_explanation(request):
    """Explain how detected emotion affects interview evaluation"""
    current_emotion = request.GET.get('emotion', validated_emotion)
    answer_text = request.GET.get('answer', '')
    category = request.GET.get('category', 'behavioral')
    
    # Emotion impact explanations
    emotion_impacts = {
        'neutral': {
            'positive': ["Shows professional composure", "Demonstrates emotional regulation", "Allows content to shine"],
            'considerations': ["May appear less enthusiastic", "Could seem disengaged if prolonged"],
            'interview_impact': "Generally positive - shows you can stay calm under pressure"
        },
        'happy': {
            'positive': ["Shows enthusiasm and passion", "Demonstrates positive attitude", "Makes good impression"],
            'considerations': ["May seem inappropriate for serious topics", "Could appear unprofessional if excessive"],
            'interview_impact': "Very positive - enthusiasm is valued by most employers"
        },
        'fear': {
            'positive': ["Shows you care about the outcome", "Normal interview nerves are understandable"],
            'considerations': ["May indicate lack of confidence", "Could affect answer quality", "Might suggest poor stress management"],
            'interview_impact': "Slightly negative - practice can help build confidence"
        },
        'angry': {
            'positive': ["May indicate passion for the topic", "Could show strong convictions"],
            'considerations': ["Often perceived negatively", "May suggest poor emotional control", "Could indicate conflict issues"],
            'interview_impact': "Generally negative - focus on staying calm and positive"
        },
        'sad': {
            'positive': ["May show empathy in appropriate contexts", "Could indicate thoughtfulness"],
            'considerations': ["Generally not ideal in interviews", "May suggest low energy", "Could indicate personal issues"],
            'interview_impact': "Usually negative - try to project more energy and optimism"
        },
        'surprise': {
            'positive': ["Shows you're engaged and thinking", "Natural response to unexpected questions"],
            'considerations': ["May indicate lack of preparation", "Could suggest difficulty adapting"],
            'interview_impact': "Neutral to slightly negative - take time to collect thoughts"
        }
    }
    
    impact_data = emotion_impacts.get(current_emotion, emotion_impacts['neutral'])
    
    # Context-specific adjustments
    if answer_text:
        word_count = len(answer_text.split())
        if word_count > 100 and current_emotion == 'neutral':
            impact_data['interview_impact'] += " Your detailed response with calm delivery shows confidence."
        elif word_count < 30 and current_emotion in ['fear', 'sad']:
            impact_data['interview_impact'] += " Brief answers with this emotion may suggest lack of preparation."
    
    # Category-specific adjustments
    if category == 'behavioral' and current_emotion == 'happy':
        impact_data['interview_impact'] += " Positive emotions work especially well for behavioral questions."
    elif category == 'technical' and current_emotion == 'neutral':
        impact_data['interview_impact'] += " Calm focus is ideal for technical discussions."
    
    return JsonResponse({
        'emotion': current_emotion,
        'impact_analysis': impact_data,
        'recommendations': get_emotion_recommendations(current_emotion, category),
        'confidence_level': float(emotion_confidence)
    })

def get_emotion_recommendations(emotion, category):
    """Get specific recommendations for managing emotions during interviews"""
    recommendations = {
        'neutral': [
            "Maintain this composed state",
            "Add slight enthusiasm when appropriate",
            "Use gestures to show engagement"
        ],
        'happy': [
            "Great energy! Maintain this positivity",
            "Ensure enthusiasm matches the question topic",
            "Balance excitement with professionalism"
        ],
        'fear': [
            "Take deep breaths to calm nerves",
            "Practice your examples beforehand",
            "Remember: some nervousness is normal",
            "Focus on your achievements and strengths"
        ],
        'angry': [
            "Take a moment to calm down",
            "Focus on positive aspects of experiences",
            "Avoid discussing negative situations",
            "Practice relaxation techniques"
        ],
        'sad': [
            "Try to project more energy",
            "Focus on positive outcomes and learnings",
            "Practice confident posture and voice",
            "Take time to prepare mentally"
        ],
        'surprise': [
            "It's okay to take a moment to think",
            "Ask for clarification if needed",
            "Practice common interview questions",
            "Stay calm and composed"
        ]
    }
    
    return recommendations.get(emotion, recommendations['neutral'])

@csrf_exempt
def reset_emotion_session(request):
    """Reset emotion session data (call when starting new interview)"""
    global emotion_session_history, user_corrections, validated_emotion, detected_emotion
    
    if request.method == "POST":
        emotion_session_history.clear()
        user_corrections.clear()
        validated_emotion = "neutral"
        detected_emotion = "neutral"
        
        # Reset emotion smoother
        reset_emotion_smoother()
        
        logger.info("Emotion session reset")
        return JsonResponse({'success': True, 'message': 'Session reset'})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def emotion_detection_accuracy(request):
    """Get emotion detection accuracy metrics - FIXED"""
    global user_corrections, emotion_session_history, _emotion_smoother
    
    total_detections = _emotion_smoother.total_frames
    total_corrections = len(emotion_session_history)
    
    if total_detections == 0:
        accuracy = 100.0
    else:
        accuracy = ((total_detections - total_corrections) / total_detections) * 100
    
    # Most commonly corrected emotions
    correction_patterns = {}
    for entry in emotion_session_history:
        pattern = f"{entry['original_emotion']} → {entry['corrected_emotion']}"
        correction_patterns[pattern] = correction_patterns.get(pattern, 0) + 1
    
    response_data = {
        'accuracy_percentage': make_json_safe(accuracy),
        'total_detections': make_json_safe(total_detections),
        'total_corrections': make_json_safe(total_corrections),
        'correction_rate': make_json_safe((total_corrections / max(total_detections, 1)) * 100),
        'common_corrections': dict(sorted(correction_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
        'session_stats': {
            'duration_minutes': make_json_safe((_emotion_smoother.total_frames * _emotion_smoother.base_sample_rate) / 60),
            'avg_confidence': make_json_safe(emotion_confidence),
            'stability_score': make_json_safe(_emotion_smoother._calculate_stability())
        }
    }
    
    return JsonResponse(response_data)

@csrf_exempt
def create_profile(request):
    """Create or update user profile during onboarding"""
    if request.method == "POST":
        try:
            # Get or create user (for demo purposes - in production, user should be authenticated)
            username = request.POST.get('username', f'user_{int(time.time())}')
            user, created = User.objects.get_or_create(username=username)
            
            # Get or update profile
            profile, _ = UserProfile.objects.get_or_create(user=user)
            
            # Update profile fields
            profile.experience_level = request.POST.get('experience_level', 'mid')
            profile.industry = request.POST.get('industry', 'technology')
            profile.role_type = request.POST.get('role_type', 'technical')
            profile.current_job_title = request.POST.get('current_job_title', '')
            profile.target_role = request.POST.get('target_role', '')
            profile.preferred_feedback_model = request.POST.get('preferred_feedback_model', 'adaptive')
            profile.learning_preference = request.POST.get('learning_preference', 'detailed')
            profile.feedback_complexity = int(request.POST.get('feedback_complexity', 3))
            profile.interview_goals = request.POST.get('interview_goals', '')
            
            # Handle JSON fields
            focus_areas = request.POST.get('focus_areas', '').split(',')
            profile.focus_areas = [area.strip() for area in focus_areas if area.strip()]
            
            weak_areas = request.POST.get('weak_areas', '').split(',')
            profile.weak_areas = [area.strip() for area in weak_areas if area.strip()]
            
            profile.onboarding_completed = True
            profile.save()
            
            # Calculate profile completion
            completion = profile.get_profile_completion()
            
            # Store user ID in session for future requests
            request.session['user_id'] = user.id
            
            return JsonResponse({
                'success': True,
                'user_id': user.id,
                'profile_completion': completion,
                'message': 'Profile created successfully'
            })
            
        except Exception as e:
            logger.error(f"Profile creation error: {e}")
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def get_profile(request):
    """Get user profile data"""
    user_id = request.session.get('user_id') or request.GET.get('user_id')
    
    if not user_id:
        return JsonResponse({'error': 'No user ID provided'})
    
    try:
        user = User.objects.get(id=user_id)
        profile = user.userprofile
        
        profile_data = {
            'user_id': user.id,
            'username': user.username,
            'experience_level': profile.experience_level,
            'industry': profile.industry,
            'role_type': profile.role_type,
            'current_job_title': profile.current_job_title,
            'target_role': profile.target_role,
            'preferred_feedback_model': profile.preferred_feedback_model,
            'learning_preference': profile.learning_preference,
            'feedback_complexity': profile.feedback_complexity,
            'interview_goals': profile.interview_goals,
            'focus_areas': profile.focus_areas,
            'weak_areas': profile.weak_areas,
            'session_count': profile.session_count,
            'onboarding_completed': profile.onboarding_completed,
            'profile_completion': profile.get_profile_completion(),
            'recommended_model': profile.get_recommended_feedback_model()
        }
        
        return JsonResponse({'success': True, 'profile': profile_data})
        
    except User.DoesNotExist:
        return JsonResponse({'error': 'User not found'})
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        return JsonResponse({'error': str(e)})

@csrf_exempt
def update_profile_preferences(request):
    """Update user preferences"""
    if request.method == "POST":
        user_id = request.session.get('user_id') or request.POST.get('user_id')
        
        if not user_id:
            return JsonResponse({'error': 'No user ID provided'})
        
        try:
            user = User.objects.get(id=user_id)
            profile = user.userprofile
            
            # Update preferences
            if 'preferred_feedback_model' in request.POST:
                profile.preferred_feedback_model = request.POST['preferred_feedback_model']
            
            if 'learning_preference' in request.POST:
                profile.learning_preference = request.POST['learning_preference']
            
            if 'feedback_complexity' in request.POST:
                profile.feedback_complexity = int(request.POST['feedback_complexity'])
            
            if 'focus_areas' in request.POST:
                focus_areas = request.POST['focus_areas'].split(',')
                profile.focus_areas = [area.strip() for area in focus_areas if area.strip()]
            
            profile.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Preferences updated successfully',
                'profile_completion': profile.get_profile_completion()
            })
            
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found'})
        except Exception as e:
            logger.error(f"Preference update error: {e}")
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

# Enhanced Feedback Functions

def get_personalized_feedback_with_profile(user_answer, question, emotion, category="behavioral"):
    """UPDATED: Use feedback models instead of basic analysis"""
    # This function should now delegate to the model-based approach
    # You'll need to pass request somehow, or refactor to get user from session differently
    
    # For now, check if we can get current request from thread-local storage
    # or pass it as parameter when calling this function
    
    try:
        # Try to get user from session (this requires request context)
        from django.contrib.sessions.models import Session
        from django.contrib.auth.models import User
        
        # This is a temporary solution - ideally refactor to pass request
        # For now, fall back to basic feedback if we can't get user context
        logger.warning("get_personalized_feedback_with_profile called without request context")
        return get_basic_feedback(user_answer, question, emotion, category)
        
    except Exception as e:
        logger.warning(f"Could not get user context: {e}")
        return get_basic_feedback(user_answer, question, emotion, category)


# ✅ VERIFY: Make sure your feedback models are imported
# Add this to the top of your views.py if not already there:
from .feedback_models import (
    FeedbackModelManager, 
    create_feedback_context,
    CoachingModel, 
    PeerReviewModel, 
    ExpertAnalysisModel, 
    StrategicModel, 
    AdaptiveModel
)
@csrf_exempt
def rate_feedback(request):
    """Allow users to rate feedback quality"""
    if request.method == "POST":
        user_id = request.session.get('user_id') or request.POST.get('user_id')
        session_id = request.POST.get('session_id')
        rating = int(request.POST.get('rating', 3))  # 1-5 scale
        notes = request.POST.get('notes', '')
        
        if not user_id:
            return JsonResponse({'error': 'No user ID provided'})
        
        try:
            user = User.objects.get(id=user_id)
            profile = user.userprofile
            
            # Find the session (latest if not specified)
            if session_id:
                session = InterviewSession.objects.get(id=session_id, user_profile=profile)
            else:
                session = InterviewSession.objects.filter(user_profile=profile).latest('created_at')
            
            # Update session with rating
            session.user_rating = rating
            session.user_notes = notes
            session.save()
            
            # Update feedback model effectiveness
            feedback_manager = get_feedback_manager()
            engagement_score = 0.8 if rating >= 4 else (0.6 if rating >= 3 else 0.3)
            feedback_manager.update_model_effectiveness(
                session.feedback_model_used, profile, rating, engagement_score
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Feedback rating saved',
                'recommended_model': profile.get_recommended_feedback_model()
            })
            
        except (User.DoesNotExist, InterviewSession.DoesNotExist) as e:
            return JsonResponse({'error': 'Session not found'})
        except Exception as e:
            logger.error(f"Rating save error: {e}")
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

@csrf_exempt
def get_feedback_models(request):
    """Get available feedback models and their descriptions"""
    feedback_manager = get_feedback_manager()
    models = feedback_manager.get_available_models()
    
    # Add effectiveness data if user is available
    user_id = request.session.get('user_id')
    if user_id:
        try:
            user = User.objects.get(id=user_id)
            profile = user.userprofile
            
            # Add effectiveness scores to model data
            for model_name in models.keys():
                effectiveness = profile.feedback_effectiveness.get(model_name, {})
                models[model_name] = {
                    'description': models[model_name],
                    'effectiveness': effectiveness.get('effectiveness', 0.5),
                    'usage_count': effectiveness.get('usage_count', 0),
                    'is_recommended': model_name == profile.get_recommended_feedback_model()
                }
        except (User.DoesNotExist, AttributeError):
            pass
    
    return JsonResponse({'models': models})

@csrf_exempt
def get_user_analytics(request):
    """Get user performance analytics"""
    user_id = request.session.get('user_id') or request.GET.get('user_id')
    
    if not user_id:
        return JsonResponse({'error': 'No user ID provided'})
    
    try:
        user = User.objects.get(id=user_id)
        profile = user.userprofile
        
        # Get recent sessions
        recent_sessions = InterviewSession.objects.filter(
            user_profile=profile
        ).order_by('-created_at')[:10]
        
        # Calculate analytics
        analytics = {
            'profile_completion': profile.get_profile_completion(),
            'total_sessions': profile.session_count,
            'average_score': 0.0,
            'improvement_trend': [],
            'category_performance': {},
            'feedback_model_effectiveness': profile.feedback_effectiveness,
            'recent_sessions': []
        }
        
        if recent_sessions:
            # Calculate average score
            total_score = sum(session.overall_score for session in recent_sessions)
            analytics['average_score'] = total_score / len(recent_sessions)
            
            # Improvement trend (last 10 sessions)
            analytics['improvement_trend'] = [
                {
                    'session': i + 1,
                    'score': session.overall_score,
                    'date': session.created_at.strftime('%Y-%m-%d')
                }
                for i, session in enumerate(reversed(recent_sessions))
            ]
            
            # Category performance
            category_scores = {}
            for session in recent_sessions:
                if session.category not in category_scores:
                    category_scores[session.category] = []
                category_scores[session.category].append(session.overall_score)
            
            analytics['category_performance'] = {
                category: sum(scores) / len(scores)
                for category, scores in category_scores.items()
            }
            
            # Recent sessions summary
            analytics['recent_sessions'] = [
                {
                    'id': session.id,
                    'category': session.category,
                    'score': session.overall_score,
                    'emotion': session.validated_emotion,
                    'date': session.created_at.strftime('%Y-%m-%d %H:%M'),
                    'feedback_model': session.feedback_model_used,
                    'user_rating': session.user_rating
                }
                for session in recent_sessions[:5]
            ]
        
        return JsonResponse({'success': True, 'analytics': analytics})
        
    except User.DoesNotExist:
        return JsonResponse({'error': 'User not found'})
    except Exception as e:
        logger.error(f"Analytics retrieval error: {e}")
        return JsonResponse({'error': str(e)})

# Updated feedback_page function to use personalized feedback
def feedback_page_with_profile(request):
    """Enhanced feedback page with personalized feedback models"""
    global stop_event, emotion_thread, validated_emotion, emotion_confidence
    
    # Stop emotion detection
    stop_event.set()
    if emotion_thread and emotion_thread.is_alive():
        emotion_thread.join(timeout=5)
    
    question = request.session.get('question', '')
    answer = request.GET.get('transcript') or request.session.get('answer', '')
    request.session['answer'] = answer
    
    category = request.session.get('category', 'behavioral')
    user_id = request.session.get('user_id')
    
    logger.info(f"Generating personalized feedback for user {user_id}")
    
    start_time = time.time()
    
    # Generate personalized feedback
    feedback = get_personalized_feedback_with_profile(answer, question, validated_emotion, category)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Get user profile for template context
    profile = None
    profile_data = {}
    if user_id:
        try:
            user = User.objects.get(id=user_id)
            profile = user.userprofile
            profile_data = {
                'experience_level': profile.get_experience_level_display(),
                'preferred_model': profile.get_preferred_feedback_model_display(),
                'session_count': profile.session_count,
                'profile_completion': profile.get_profile_completion()
            }
        except (User.DoesNotExist, AttributeError):
            pass
    
    # Get additional metrics
    vectorizer = get_vectorizer()
    metrics = {}
    if vectorizer:
        try:
            semantic_score = vectorizer.analyze_answer_semantics(answer, category, profile.experience_level if profile else 'mid')
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
        'emotion': validated_emotion,
        'raw_emotion': detected_emotion,
        'emotion_confidence': emotion_confidence,
        'feedback': feedback,
        'feedback_type': "Personalized" if profile else "Standard",
        'category': category,
        'camera_available': camera_available,
        'processing_time': f"{processing_time:.1f}s",
        'metrics': metrics,
        'vectorizer_available': vectorizer is not None,
        'profile_data': profile_data,
        'user_id': user_id,
        'show_rating': True  # Enable feedback rating
    })

# Helper function to get or create anonymous user session
def get_or_create_user_session(request):  # Make sure request is passed
    """Get or create user session for anonymous usage"""
    user_id = request.session.get('user_id')
    
    if not user_id:
        # Create anonymous user
        username = f'anonymous_{int(time.time())}_{random.randint(1000, 9999)}'
        user = User.objects.create(username=username)
        request.session['user_id'] = user.id
        
        # Create profile with default values
        profile, created = UserProfile.objects.get_or_create(user=user)
        if created:
            profile.experience_level = 'mid'
            profile.preferred_feedback_model = 'adaptive'
            profile.save()
        
        logger.info(f"Created anonymous user session: {user.id}")
        return user.id
    
    return user_id
def analytics_page(request):
    return render(request, 'analytics.html')

@csrf_exempt
def debug_metrics(request):
    """Debug endpoint to test metrics generation"""
    answer = request.GET.get('answer', 'This is a sample answer for testing metrics generation.')
    question = request.GET.get('question', 'Tell me about a challenge you faced.')
    category = request.GET.get('category', 'behavioral')
    
    try:
        # Get vectorizer
        vectorizer = get_vectorizer()
        
        if vectorizer:
            # Test semantic analysis
            semantic_score = vectorizer.analyze_answer_semantics(answer, category, 'mid')
            
            metrics = {
                'content_coverage': make_json_safe(semantic_score.content_coverage),
                'depth_score': make_json_safe(semantic_score.depth_score),
                'relevance_score': make_json_safe(semantic_score.relevance_score),
                'technical_depth': make_json_safe(semantic_score.technical_depth),
                'leadership_indicators': make_json_safe(semantic_score.leadership_indicators),
                'communication_clarity': make_json_safe(semantic_score.communication_clarity),
                'emotion_confidence': make_json_safe(emotion_confidence)
            }
            
            return JsonResponse({
                'success': True,
                'vectorizer_available': True,
                'metrics': metrics,
                'raw_semantic_score': {
                    'content_coverage': float(semantic_score.content_coverage),
                    'strengths': semantic_score.strengths,
                    'gaps': semantic_score.gaps
                }
            })
        else:
            return JsonResponse({
                'success': False,
                'vectorizer_available': False,
                'error': 'Vectorizer not available',
                'fallback_metrics': {
                    'word_count': len(answer.split()),
                    'emotion_confidence': make_json_safe(emotion_confidence)
                }
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': str(e.__class__.__name__)
        })
    
# Add these new API endpoints to your views.py

@csrf_exempt
def get_session_analytics(request):
    """Get comprehensive session analytics from database"""
    user_id = request.session.get('user_id') or request.GET.get('user_id')
    
    try:
        # Get or create anonymous user if none exists
        if not user_id:
            user_id = get_or_create_user_session(request)
        
        user = User.objects.get(id=user_id)
        profile = user.userprofile
        
        # Get all sessions for this user
        sessions = InterviewSession.objects.filter(user_profile=profile).order_by('-created_at')
        
        # Calculate analytics
        total_sessions = sessions.count()
        
        # Average scores
        if total_sessions > 0:
            avg_overall_score = sessions.aggregate(avg_score=models.Avg('overall_score'))['avg_score'] or 0
            avg_content_score = sessions.aggregate(avg_score=models.Avg('content_score'))['avg_score'] or 0
            avg_emotion_score = sessions.aggregate(avg_score=models.Avg('emotion_score'))['avg_score'] or 0
            avg_user_rating = sessions.filter(user_rating__isnull=False).aggregate(avg_rating=models.Avg('user_rating'))['avg_rating'] or 0
        else:
            avg_overall_score = avg_content_score = avg_emotion_score = avg_user_rating = 0
        
        # Category performance
        category_performance = {}
        categories = sessions.values('category').distinct()
        for cat in categories:
            category = cat['category']
            cat_sessions = sessions.filter(category=category)
            cat_avg = cat_sessions.aggregate(avg_score=models.Avg('overall_score'))['avg_score'] or 0
            category_performance[category] = cat_avg
        
        # Recent trend (last 10 sessions)
        recent_sessions = sessions[:10]
        improvement_trend = []
        for i, session in enumerate(reversed(recent_sessions)):
            improvement_trend.append({
                'session': i + 1,
                'score': float(session.overall_score),
                'date': session.created_at.strftime('%Y-%m-%d'),
                'category': session.category,
                'emotion': session.validated_emotion
            })
        
        # Emotion accuracy (based on confidence and corrections)
        emotion_stats = {}
        emotion_data = sessions.values('detected_emotion', 'validated_emotion', 'emotion_confidence')
        
        for data in emotion_data:
            detected = data['detected_emotion']
            validated = data['validated_emotion']
            confidence = data['emotion_confidence'] or 0.5
            
            if detected not in emotion_stats:
                emotion_stats[detected] = {'total': 0, 'accurate': 0, 'avg_confidence': 0}
            
            emotion_stats[detected]['total'] += 1
            emotion_stats[detected]['avg_confidence'] += confidence
            
            if detected == validated:
                emotion_stats[detected]['accurate'] += 1
        
        # Calculate accuracy percentages
        emotion_accuracy = {}
        for emotion, stats in emotion_stats.items():
            if stats['total'] > 0:
                accuracy = stats['accurate'] / stats['total']
                avg_conf = stats['avg_confidence'] / stats['total']
                emotion_accuracy[emotion] = {
                    'accuracy': accuracy,
                    'confidence': avg_conf,
                    'total_detections': stats['total']
                }
        
        # Recent sessions data
        recent_sessions_data = []
        for session in recent_sessions:
            recent_sessions_data.append({
                'id': session.id,
                'category': session.category,
                'overall_score': float(session.overall_score),
                'content_score': float(session.content_score),
                'emotion_score': float(session.emotion_score),
                'detected_emotion': session.detected_emotion,
                'validated_emotion': session.validated_emotion,
                'emotion_confidence': float(session.emotion_confidence),
                'feedback_model_used': session.feedback_model_used,
                'user_rating': session.user_rating,
                'created_at': session.created_at.isoformat(),
                'question': session.question[:100] + '...' if len(session.question) > 100 else session.question,
                'answer_length': len(session.answer.split()) if session.answer else 0
            })
        
        # Get feedback model effectiveness
        model_effectiveness = profile.feedback_effectiveness
        
        # Profile completion
        profile_completion = profile.get_profile_completion() if hasattr(profile, 'get_profile_completion') else 0.5
        
        analytics_data = {
            'success': True,
            'user_id': user_id,
            'profile_data': {
                'username': user.username,
                'experience_level': profile.experience_level,
                'industry': profile.industry,
                'role_type': profile.role_type,
                'session_count': profile.session_count,
                'profile_completion': profile_completion
            },
            'summary_stats': {
                'total_sessions': total_sessions,
                'average_overall_score': float(avg_overall_score),
                'average_content_score': float(avg_content_score),
                'average_emotion_score': float(avg_emotion_score),
                'average_user_rating': float(avg_user_rating)
            },
            'category_performance': {k: float(v) for k, v in category_performance.items()},
            'emotion_accuracy': emotion_accuracy,
            'improvement_trend': improvement_trend,
            'recent_sessions': recent_sessions_data,
            'feedback_model_effectiveness': model_effectiveness,
            'recommendations': generate_personalized_recommendations(profile, sessions, category_performance, emotion_accuracy)
        }
        
        return JsonResponse(analytics_data)
        
    except User.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'User not found'})
    except Exception as e:
        logger.error(f"Analytics retrieval error: {e}")
        return JsonResponse({'success': False, 'error': str(e)})

def generate_personalized_recommendations(profile, sessions, category_performance, emotion_accuracy):
    """Generate personalized recommendations based on user data"""
    recommendations = []
    
    total_sessions = sessions.count()
    
    # Session frequency recommendations
    if total_sessions == 0:
        recommendations.append({
            'icon': '🚀',
            'title': 'Start Your Journey',
            'description': 'Take your first mock interview to begin building your skills and confidence.',
            'priority': 'high',
            'action': 'start_interview'
        })
    elif total_sessions < 5:
        recommendations.append({
            'icon': '🔄',
            'title': 'Build Consistency',
            'description': 'Practice regularly to see improvement. Try to complete 2-3 sessions per week.',
            'priority': 'medium',
            'action': 'increase_frequency'
        })
    
    # Performance-based recommendations
    if total_sessions > 0:
        avg_score = sessions.aggregate(avg=models.Avg('overall_score'))['avg'] or 0
        
        if avg_score < 0.6:
            recommendations.append({
                'icon': '📚',
                'title': 'Focus on Fundamentals',
                'description': 'Your scores suggest focusing on basic interview techniques and the STAR method.',
                'priority': 'high',
                'action': 'study_fundamentals'
            })
        elif avg_score >= 0.8:
            recommendations.append({
                'icon': '🎯',
                'title': 'Advanced Practice',
                'description': 'Excellent progress! Try more challenging scenarios and leadership questions.',
                'priority': 'low',
                'action': 'advanced_practice'
            })
    
    # Category-specific recommendations
    if category_performance:
        lowest_category = min(category_performance.items(), key=lambda x: x[1])
        if lowest_category[1] < 0.6:
            recommendations.append({
                'icon': '🎯',
                'title': f'Improve {lowest_category[0].title()} Skills',
                'description': f'Focus on practicing more {lowest_category[0]} questions to strengthen this area.',
                'priority': 'medium',
                'action': f'practice_{lowest_category[0]}'
            })
    
    # Emotion accuracy recommendations
    if emotion_accuracy:
        low_accuracy_emotions = [
            emotion for emotion, data in emotion_accuracy.items()
            if data['accuracy'] < 0.7 and data['total_detections'] >= 3
        ]
        
        if low_accuracy_emotions:
            recommendations.append({
                'icon': '🎭',
                'title': 'Emotion Recognition Training',
                'description': f'Practice managing your emotions during {", ".join(low_accuracy_emotions)} responses.',
                'priority': 'medium',
                'action': 'emotion_training'
            })
    
    # Profile completion recommendation
    profile_completion = profile.get_profile_completion() if hasattr(profile, 'get_profile_completion') else 0.5
    if profile_completion < 0.8:
        recommendations.append({
            'icon': '👤',
            'title': 'Complete Your Profile',
            'description': 'A complete profile helps provide more personalized feedback and recommendations.',
            'priority': 'low',
            'action': 'complete_profile'
        })
    
    # Sort by priority
    priority_order = {'high': 3, 'medium': 2, 'low': 1}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
    
    return recommendations[:6]  # Return top 6 recommendations

@csrf_exempt
def record_session_data(request):
    """Record session data for analytics tracking"""
    if request.method == "POST":
        try:
            user_id = request.session.get('user_id')
            if not user_id:
                user_id = get_or_create_user_session(request)
                request.session['user_id'] = user_id
            
            user = User.objects.get(id=user_id)
            profile = user.userprofile
            
            # Get session data
            session_data = {
                'category': request.POST.get('category', 'behavioral'),
                'question': request.POST.get('question', ''),
                'answer': request.POST.get('answer', ''),
                'overall_score': float(request.POST.get('overall_score', 0.6)),
                'content_score': float(request.POST.get('content_score', 0.6)),
                'emotion_score': float(request.POST.get('emotion_score', 0.6)),
                'detected_emotion': request.POST.get('detected_emotion', 'neutral'),
                'validated_emotion': request.POST.get('validated_emotion', 'neutral'),
                'emotion_confidence': float(request.POST.get('emotion_confidence', 0.5)),
                'feedback_model_used': request.POST.get('feedback_model_used', 'basic'),
                'feedback_content': request.POST.get('feedback_content', ''),
                'vectorizer_used': request.POST.get('vectorizer_used', 'false').lower() == 'true'
            }
            
            # Create session record
            session = InterviewSession.objects.create(
                user_profile=profile,
                **session_data
            )
            
            # Update profile session count
            profile.session_count += 1
            profile.save()
            
            logger.info(f"Session recorded for user {user_id}: {session.id}")
            
            return JsonResponse({
                'success': True,
                'session_id': session.id,
                'message': 'Session recorded successfully'
            })
            
        except Exception as e:
            logger.error(f"Session recording error: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def clear_user_analytics(request):
    """Clear all analytics data for the current user"""
    if request.method == "POST":
        user_id = request.session.get('user_id') or request.POST.get('user_id')
        
        if not user_id:
            return JsonResponse({'success': False, 'error': 'No user ID provided'})
        
        try:
            user = User.objects.get(id=user_id)
            profile = user.userprofile
            
            # Delete all sessions
            deleted_count = InterviewSession.objects.filter(user_profile=profile).delete()[0]
            
            # Reset profile counters
            profile.session_count = 0
            profile.performance_history = []
            profile.feedback_effectiveness = {}
            profile.save()
            
            logger.info(f"Cleared {deleted_count} sessions for user {user_id}")
            
            return JsonResponse({
                'success': True,
                'message': f'Cleared {deleted_count} sessions successfully'
            })
            
        except User.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'User not found'})
        except Exception as e:
            logger.error(f"Clear analytics error: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

# Add this import at the top of your views.py


def feedback_page_enhanced_with_recording(request):
    """Enhanced feedback page that actually uses the feedback models"""
    global stop_event, emotion_thread, validated_emotion, emotion_confidence
    
    # Stop emotion detection
    stop_event.set()
    if emotion_thread and emotion_thread.is_alive():
        emotion_thread.join(timeout=5)
    
    question = request.session.get('question', '')
    answer = request.GET.get('transcript') or request.session.get('answer', '')
    request.session['answer'] = answer
    
    category = request.session.get('category', 'behavioral')
    role_level = request.session.get('role_level', 'mid')
    
    logger.info(f"Generating feedback with validated emotion: {validated_emotion} (confidence: {emotion_confidence:.3f})")
    
    start_time = time.time()
    
    # ✅ NOW ACTUALLY USE THE FEEDBACK MODELS
    feedback = get_personalized_feedback_using_models(
        answer, question, validated_emotion, category, role_level, request
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate scores for recording
    vectorizer = get_vectorizer()
    overall_score = 0.6
    content_score = 0.6
    
    if vectorizer:
        try:
            enhanced_analysis = enhanced_analyze_answer_content(
                answer, question, category, role_level
            )
            overall_score = enhanced_analysis.get('overall_score', 0.6)
            semantic_analysis = enhanced_analysis.get('semantic_analysis')
            if semantic_analysis:
                content_score = semantic_analysis.get('content_coverage', 0.6)
        except Exception as e:
            logger.warning(f"Could not calculate scores: {e}")
    
    emotion_score = emotion_confidence
    
    # Record session data
    try:
        user_id = request.session.get('user_id')
        if not user_id:
            user_id = get_or_create_user_session(request)
            request.session['user_id'] = user_id
        
        user = User.objects.get(id=user_id)
        profile = user.userprofile
        
        # Create session record
        session = InterviewSession.objects.create(
            user_profile=profile,
            category=category,
            question=question,
            answer=answer,
            overall_score=overall_score,
            content_score=content_score,
            emotion_score=emotion_score,
            detected_emotion=detected_emotion,
            validated_emotion=validated_emotion,
            emotion_confidence=emotion_confidence,
            feedback_model_used=profile.preferred_feedback_model,
            feedback_content=feedback,
            vectorizer_used=vectorizer is not None
        )
        
        # Update profile
        profile.session_count += 1
        profile.save()
        
        logger.info(f"Session recorded: {session.id}")
        
    except Exception as e:
        logger.error(f"Failed to record session: {e}")
    
    # Generate metrics for display
    metrics = {
        'overall_score': f"{overall_score:.1%}",
        'content_coverage': f"{content_score:.1%}",
        'emotion_confidence': f"{emotion_confidence:.1%}",
        'processing_time': f"{processing_time:.1f}s"
    }
    
    return render(request, 'feedback.html', {
        'question': question,
        'answer': answer,
        'emotion': validated_emotion,
        'raw_emotion': detected_emotion,
        'emotion_confidence': emotion_confidence,
        'feedback': feedback,
        'feedback_type': "Personalized Model-Based",
        'category': category,
        'role_level': role_level,
        'camera_available': camera_available,
        'processing_time': f"{processing_time:.1f}s",
        'metrics': metrics,
        'vectorizer_available': vectorizer is not None,
        'show_rating': True,
        'user_id': request.session.get('user_id'),
        'user_profile': profile if 'profile' in locals() else None,
        'feedback_manager_active': True,  # Since we're now using it
        'validator_available': True,
    })


def get_personalized_feedback_using_models(user_answer, question, emotion, category, role_level, request):
    """NEW: Generate feedback using the actual feedback models"""
    
    # Get or create user profile
    user_id = request.session.get('user_id')
    if not user_id:
        user_id = get_or_create_user_session(request)
        request.session['user_id'] = user_id
    
    try:
        user = User.objects.get(id=user_id)
        profile = user.userprofile
    except (User.DoesNotExist, AttributeError):
        # Fallback to basic feedback if no profile
        logger.warning("No user profile found, using basic feedback")
        return get_basic_feedback(user_answer, question, emotion, category)
    
    # Get enhanced analysis
    enhanced_analysis = enhanced_analyze_answer_content(
        user_answer, question, category, profile.experience_level
    )
    
    # Create feedback context
    context = create_feedback_context(
        user_profile=profile,
        session_data={
            'category': category,
            'question': question,
            'answer': user_answer,
            'timestamp': timezone.now().isoformat()
        },
        performance_scores={
            'overall_score': enhanced_analysis.get('overall_score', 0.6),
            'content_score': enhanced_analysis.get('semantic_analysis', {}).get('content_coverage', 0.6) if enhanced_analysis.get('semantic_analysis') else 0.6,
            'emotion_score': emotion_confidence,
        },
        emotion_data={
            'detected_emotion': detected_emotion,
            'validated_emotion': validated_emotion,
            'confidence': emotion_confidence,
            'stability_score': _emotion_smoother._calculate_stability() if _emotion_smoother else 0.5
        },
        content_analysis=enhanced_analysis
    )
    
    # ✅ ACTUALLY USE THE FEEDBACK MODELS
    feedback_manager = get_feedback_manager()
    feedback = feedback_manager.generate_feedback(profile.preferred_feedback_model, context)
    
    logger.info(f"Generated feedback using model: {profile.preferred_feedback_model}")
    
    return feedback
