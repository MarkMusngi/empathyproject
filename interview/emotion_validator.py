# emotion_validator.py - Cross-validation system for emotion detection
import logging
import re
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

# Lightweight sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("vaderSentiment not available, using fallback sentiment analysis")

logger = logging.getLogger(__name__)

@dataclass
class EmotionValidation:
    """Results of emotion-sentiment cross-validation"""
    original_emotion: str
    validated_emotion: str
    confidence_score: float  # 0.0 to 1.0
    sentiment_score: float   # -1.0 to 1.0
    correlation_strength: float  # 0.0 to 1.0
    contextual_interpretation: str
    flags: list  # Potential issues detected
    adjustment_reason: str

class EmotionValidator:
    """Cross-validation system for facial emotion vs. answer sentiment"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_history = deque(maxlen=10)  # Track recent emotions
        self.validation_stats = defaultdict(int)
        self.false_positive_patterns = []
        
        # Initialize sentiment analyzer
        self._initialize_sentiment_analyzer()
        
        # Emotion-sentiment correlation mappings
        self.emotion_sentiment_map = {
            'happy': {'positive': 0.9, 'neutral': 0.3, 'negative': 0.1},
            'neutral': {'positive': 0.4, 'neutral': 0.8, 'negative': 0.4},
            'sad': {'positive': 0.1, 'neutral': 0.3, 'negative': 0.9},
            'angry': {'positive': 0.1, 'neutral': 0.2, 'negative': 0.8},
            'fear': {'positive': 0.2, 'neutral': 0.4, 'negative': 0.7},
            'surprise': {'positive': 0.6, 'neutral': 0.5, 'negative': 0.3},
            'disgust': {'positive': 0.1, 'neutral': 0.2, 'negative': 0.8}
        }
        
        # Contextual interpretation rules
        self.context_rules = {
            ('happy', 'negative'): "Might be nervous laughter or forced positivity",
            ('neutral', 'positive'): "Confident and composed delivery",
            ('neutral', 'negative'): "Professional but discussing challenges",
            ('sad', 'positive'): "Possible misdetection - may be focused/serious",
            ('angry', 'neutral'): "Might be intense focus rather than anger",
            ('fear', 'positive'): "Nervous but optimistic about the topic"
        }
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis with fallback"""
        try:
            if VADER_AVAILABLE:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            else:
                logger.info("Using fallback sentiment analysis")
        except Exception as e:
            logger.warning(f"Sentiment analyzer initialization failed: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of answer text"""
        if not text or len(text.strip()) < 5:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        if self.sentiment_analyzer and VADER_AVAILABLE:
            return self._vader_sentiment(text)
        else:
            return self._fallback_sentiment(text)
    
    def _vader_sentiment(self, text: str) -> Dict[str, float]:
        """VADER sentiment analysis"""
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis using keyword matching"""
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = [
            'good', 'great', 'excellent', 'successful', 'achieved', 'improved',
            'positive', 'excited', 'proud', 'accomplished', 'effective',
            'solution', 'resolved', 'beneficial', 'opportunity', 'growth'
        ]
        
        # Negative indicators
        negative_words = [
            'difficult', 'challenging', 'problem', 'issue', 'failed', 'mistake',
            'conflict', 'disagreement', 'struggle', 'obstacle', 'setback',
            'frustrated', 'disappointed', 'concerned', 'worried', 'stress'
        ]
        
        # Count occurrences
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Normalize by text length
        word_count = len(text.split())
        pos_score = min(pos_count / max(word_count / 20, 1), 1.0)
        neg_score = min(neg_count / max(word_count / 20, 1), 1.0)
        neu_score = max(0.0, 1.0 - pos_score - neg_score)
        
        # Calculate compound score
        compound = pos_score - neg_score
        
        return {
            'compound': compound,
            'pos': pos_score,
            'neu': neu_score,
            'neg': neg_score
        }
    
    def classify_sentiment(self, sentiment_scores: Dict[str, float]) -> str:
        """Classify sentiment as positive, negative, or neutral"""
        compound = sentiment_scores.get('compound', 0.0)
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_emotion_sentiment_correlation(self, emotion: str, sentiment: str) -> float:
        """Calculate correlation strength between detected emotion and sentiment"""
        emotion_map = self.emotion_sentiment_map.get(emotion.lower(), {})
        correlation = emotion_map.get(sentiment, 0.2)  # Default low correlation
        return correlation
    
    def detect_false_positive_patterns(self, emotion: str, text: str, 
                                     sentiment_scores: Dict[str, float]) -> list:
        """Detect patterns that might indicate false positive emotion detection"""
        flags = []
        text_lower = text.lower()
        
        # Pattern 1: Technical discussion detected as emotion
        technical_indicators = ['system', 'code', 'database', 'algorithm', 'api', 
                              'implementation', 'architecture', 'framework']
        if any(indicator in text_lower for indicator in technical_indicators):
            if emotion in ['angry', 'disgust']:
                flags.append("Technical discussion may be misread as negative emotion")
        
        # Pattern 2: Confident discussion of challenges
        challenge_words = ['challenge', 'problem', 'difficult', 'obstacle']
        solution_words = ['solved', 'resolved', 'overcame', 'managed', 'handled']
        if (any(word in text_lower for word in challenge_words) and 
            any(word in text_lower for word in solution_words)):
            if emotion == 'sad':
                flags.append("Problem-solving discussion may appear sad but shows competence")
        
        # Pattern 3: High engagement misread as anger
        engagement_words = ['passionate', 'excited', 'enthusiastic', 'love', 'enjoy']
        if any(word in text_lower for word in engagement_words) and emotion == 'angry':
            flags.append("High engagement may be misread as anger")
        
        # Pattern 4: Concentration face misread as negative
        thinking_indicators = ['analyzed', 'considered', 'evaluated', 'thought', 'decided']
        if any(word in text_lower for word in thinking_indicators) and emotion in ['angry', 'sad']:
            flags.append("Concentration expression may be misread as negative emotion")
        
        # Pattern 5: Sentiment-emotion mismatch
        sentiment = self.classify_sentiment(sentiment_scores)
        correlation = self.calculate_emotion_sentiment_correlation(emotion, sentiment)
        if correlation < 0.3:
            flags.append(f"Low correlation between {emotion} emotion and {sentiment} sentiment")
        
        return flags
    
    def get_contextual_interpretation(self, emotion: str, sentiment: str, 
                                    text: str, correlation: float) -> str:
        """Provide contextual interpretation of emotion in interview context"""
        
        # Check predefined context rules
        context_key = (emotion.lower(), sentiment)
        if context_key in self.context_rules:
            base_interpretation = self.context_rules[context_key]
        else:
            base_interpretation = f"Standard {emotion} emotion with {sentiment} content"
        
        # Add correlation context
        if correlation >= 0.7:
            confidence = "high confidence"
        elif correlation >= 0.4:
            confidence = "moderate confidence"
        else:
            confidence = "low confidence - possible misdetection"
        
        # Add interview-specific context
        text_lower = text.lower()
        if 'team' in text_lower or 'led' in text_lower:
            context_addition = " during leadership discussion"
        elif any(word in text_lower for word in ['challenge', 'problem', 'difficult']):
            context_addition = " while discussing challenges"
        elif any(word in text_lower for word in ['success', 'achieved', 'accomplished']):
            context_addition = " while sharing achievements"
        else:
            context_addition = ""
        
        return f"{base_interpretation}{context_addition} ({confidence})"
    
    def validate_emotion(self, detected_emotion: str, answer_text: str, 
                        speaking_duration: float = 0) -> EmotionValidation:
        """Main validation function - cross-validate emotion with sentiment"""
        
        start_time = time.time()
        
        # Analyze sentiment
        sentiment_scores = self.analyze_text_sentiment(answer_text)
        sentiment_class = self.classify_sentiment(sentiment_scores)
        
        # Calculate correlation
        correlation = self.calculate_emotion_sentiment_correlation(detected_emotion, sentiment_class)
        
        # Detect potential false positives
        flags = self.detect_false_positive_patterns(detected_emotion, answer_text, sentiment_scores)
        
        # Determine confidence score
        base_confidence = correlation
        
        # Adjust confidence based on flags
        confidence_penalty = min(len(flags) * 0.15, 0.4)
        confidence_score = max(0.1, base_confidence - confidence_penalty)
        
        # Determine validated emotion
        validated_emotion = detected_emotion
        adjustment_reason = "No adjustment needed"
        
        # Apply corrections for low confidence cases
        if confidence_score < 0.3 and flags:
            if "Technical discussion" in str(flags) and detected_emotion in ['angry', 'disgust']:
                validated_emotion = 'neutral'
                adjustment_reason = "Technical discussion context"
            elif "Problem-solving discussion" in str(flags) and detected_emotion == 'sad':
                validated_emotion = 'neutral'
                adjustment_reason = "Problem-solving confidence"
            elif "High engagement" in str(flags) and detected_emotion == 'angry':
                validated_emotion = 'happy'
                adjustment_reason = "Engagement misclassification"
            elif "Concentration expression" in str(flags):
                validated_emotion = 'neutral'
                adjustment_reason = "Concentration face correction"
        
        # Get contextual interpretation
        contextual_interpretation = self.get_contextual_interpretation(
            validated_emotion, sentiment_class, answer_text, correlation
        )
        
        # Update statistics
        self.validation_stats['total_validations'] += 1
        if validated_emotion != detected_emotion:
            self.validation_stats['adjustments_made'] += 1
        if flags:
            self.validation_stats['flags_detected'] += 1
        
        # Store in history
        self.emotion_history.append({
            'emotion': validated_emotion,
            'confidence': confidence_score,
            'timestamp': time.time()
        })
        
        processing_time = time.time() - start_time
        logger.info(f"Emotion validation completed in {processing_time:.3f}s: "
                   f"{detected_emotion} -> {validated_emotion} (confidence: {confidence_score:.3f})")
        
        return EmotionValidation(
            original_emotion=detected_emotion,
            validated_emotion=validated_emotion,
            confidence_score=confidence_score,
            sentiment_score=sentiment_scores.get('compound', 0.0),
            correlation_strength=correlation,
            contextual_interpretation=contextual_interpretation,
            flags=flags,
            adjustment_reason=adjustment_reason
        )
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics for monitoring"""
        total = self.validation_stats.get('total_validations', 1)
        return {
            'total_validations': total,
            'adjustment_rate': self.validation_stats.get('adjustments_made', 0) / total,
            'flag_rate': self.validation_stats.get('flags_detected', 0) / total,
            'recent_emotions': [h['emotion'] for h in list(self.emotion_history)[-5:]],
            'avg_recent_confidence': sum(h['confidence'] for h in self.emotion_history) / max(len(self.emotion_history), 1)
        }
    
    def update_user_feedback(self, original_emotion: str, user_corrected_emotion: str):
        """Update system based on user feedback for continuous improvement"""
        # Log the correction for future model improvements
        logger.info(f"User correction: {original_emotion} -> {user_corrected_emotion}")
        
        # Update false positive patterns (simple implementation)
        if original_emotion != user_corrected_emotion:
            self.validation_stats['user_corrections'] = self.validation_stats.get('user_corrections', 0) + 1
            
            # Add to false positive patterns for future reference
            correction_pattern = {
                'original': original_emotion,
                'corrected': user_corrected_emotion,
                'timestamp': time.time()
            }
            self.false_positive_patterns.append(correction_pattern)
            
            # Keep only recent patterns (last 50)
            if len(self.false_positive_patterns) > 50:
                self.false_positive_patterns = self.false_positive_patterns[-50:]

# Global validator instance
_validator_instance = None

def get_emotion_validator():
    """Singleton pattern for emotion validator"""
    global _validator_instance
    if _validator_instance is None:
        try:
            _validator_instance = EmotionValidator()
            logger.info("Emotion validator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize emotion validator: {e}")
            _validator_instance = None
    return _validator_instance