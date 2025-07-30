# content_vectorizer.py - Advanced semantic analysis for interview answers
import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle
import hashlib

# Lightweight NLP imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using fallback methods")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic similarity")

logger = logging.getLogger(__name__)

@dataclass
class SemanticScore:
    """Structured semantic analysis results"""
    content_coverage: float  # How well the answer covers expected themes
    depth_score: float      # Level of detail and specificity
    relevance_score: float  # Alignment with question intent
    leadership_indicators: float
    problem_solving_indicators: float
    technical_depth: float
    communication_clarity: float
    gaps: List[str]         # Missing semantic elements
    strengths: List[str]    # Strong semantic elements

class ContentVectorizer:
    """Main vectorization system for interview answer analysis"""
    
    def __init__(self, cache_dir="vector_cache"):
        self.cache_dir = cache_dir
        self.model = None
        self.tfidf_vectorizer = None
        self.ideal_patterns = {}
        self.theme_vectors = {}
        self.fallback_mode = False
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize the model
        self._initialize_model()
        
        # Load or create ideal response patterns
        self._load_ideal_patterns()
    
    def _initialize_model(self):
        """Initialize semantic model with fallbacks"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use a lightweight, fast model
                model_name = "all-MiniLM-L6-v2"  # 384 dimensions, very fast
                cache_path = os.path.join(self.cache_dir, "sentence_transformer")
                
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name, cache_folder=cache_path)
                logger.info("Sentence transformer model loaded successfully")
                
            elif SKLEARN_AVAILABLE:
                logger.info("Using TF-IDF fallback for vectorization")
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=500,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )
                self.fallback_mode = True
                
            else:
                logger.warning("No vectorization libraries available, using basic analysis")
                self.fallback_mode = True
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}, falling back to basic analysis")
            self.fallback_mode = True
    
    def _load_ideal_patterns(self):
        """Load or create ideal response patterns for different categories"""
        patterns_file = os.path.join(self.cache_dir, "ideal_patterns.json")
        
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    self.ideal_patterns = json.load(f)
                logger.info("Loaded cached ideal patterns")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached patterns: {e}")
        
        # Create ideal patterns for different question types and roles
        self.ideal_patterns = {
            'behavioral': {
                'leadership': [
                    "I led a team of developers through a challenging project deadline",
                    "I motivated team members by recognizing their contributions and providing clear direction",
                    "I delegated tasks based on each person's strengths while maintaining accountability",
                    "I facilitated team meetings to ensure everyone's voice was heard and conflicts were resolved",
                    "I mentored junior colleagues and helped them develop their technical skills"
                ],
                'problem_solving': [
                    "I analyzed the root cause by gathering data from multiple sources and stakeholders",
                    "I evaluated several alternative solutions considering time, cost, and technical constraints",
                    "I broke down the complex problem into smaller, manageable components",
                    "I collaborated with cross-functional teams to implement a comprehensive solution",
                    "I measured the results and adjusted the approach based on feedback and outcomes"
                ],
                'communication': [
                    "I presented complex technical concepts to non-technical stakeholders using analogies",
                    "I actively listened to concerns and asked clarifying questions to understand different perspectives",
                    "I documented the process clearly for future reference and team knowledge sharing",
                    "I provided regular updates to keep all stakeholders informed of progress and challenges",
                    "I facilitated productive discussions between conflicting parties to reach consensus"
                ]
            },
            'technical': {
                'system_design': [
                    "I designed a scalable architecture considering current needs and future growth",
                    "I evaluated trade-offs between performance, maintainability, and development speed",
                    "I implemented monitoring and logging to ensure system reliability and debugging capability",
                    "I chose appropriate technologies based on team expertise and project requirements",
                    "I documented the system design and created runbooks for operational support"
                ],
                'coding_practices': [
                    "I wrote clean, readable code following established patterns and conventions",
                    "I implemented comprehensive unit tests to ensure code reliability and prevent regressions",
                    "I conducted thorough code reviews focusing on logic, security, and performance",
                    "I refactored legacy code to improve maintainability while preserving functionality",
                    "I optimized database queries and application performance using profiling tools"
                ]
            },
            'situational': {
                'conflict_resolution': [
                    "I would first listen to all parties to understand their perspectives and concerns",
                    "I would identify common ground and shared objectives to build upon",
                    "I would facilitate a structured discussion to address specific issues constructively",
                    "I would propose solutions that balance different needs and priorities fairly",
                    "I would establish clear agreements and follow-up processes to prevent future conflicts"
                ],
                'priority_management': [
                    "I would assess the urgency and impact of each task using a structured framework",
                    "I would communicate with stakeholders to clarify expectations and negotiate deadlines",
                    "I would break down large tasks into smaller milestones to track progress effectively",
                    "I would delegate appropriate tasks to team members based on their skills and availability",
                    "I would regularly review and adjust priorities based on changing business needs"
                ]
            }
        }
        
        # Cache the patterns
        try:
            with open(patterns_file, 'w') as f:
                json.dump(self.ideal_patterns, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache ideal patterns: {e}")
        
        # Pre-compute vectors for ideal patterns
        self._compute_ideal_vectors()
    
    def _compute_ideal_vectors(self):
        """Pre-compute vectors for ideal response patterns"""
        if self.fallback_mode or not self.model:
            return
        
        vectors_file = os.path.join(self.cache_dir, "ideal_vectors.pkl")
        
        try:
            # Try to load cached vectors
            if os.path.exists(vectors_file):
                with open(vectors_file, 'rb') as f:
                    self.theme_vectors = pickle.load(f)
                logger.info("Loaded cached ideal vectors")
                return
        except Exception as e:
            logger.warning(f"Failed to load cached vectors: {e}")
        
        # Compute vectors for all ideal patterns
        logger.info("Computing vectors for ideal patterns...")
        self.theme_vectors = {}
        
        for category, themes in self.ideal_patterns.items():
            self.theme_vectors[category] = {}
            for theme, examples in themes.items():
                try:
                    vectors = self.model.encode(examples)
                    # Store mean vector for the theme
                    self.theme_vectors[category][theme] = np.mean(vectors, axis=0)
                except Exception as e:
                    logger.warning(f"Failed to compute vector for {category}/{theme}: {e}")
        
        # Cache the computed vectors
        try:
            with open(vectors_file, 'wb') as f:
                pickle.dump(self.theme_vectors, f)
            logger.info("Cached ideal vectors successfully")
        except Exception as e:
            logger.warning(f"Failed to cache vectors: {e}")
    
    def analyze_answer_semantics(self, answer: str, question_category: str, 
                                target_role_level: str = "mid") -> SemanticScore:
        """
        Main semantic analysis function
        
        Args:
            answer: User's interview answer
            question_category: 'behavioral', 'technical', 'situational', 'motivational'
            target_role_level: 'junior', 'mid', 'senior', 'lead'
        
        Returns:
            SemanticScore with detailed analysis
        """
        
        if not answer or len(answer.strip()) < 10:
            return SemanticScore(
                content_coverage=0.0,
                depth_score=0.0,
                relevance_score=0.0,
                leadership_indicators=0.0,
                problem_solving_indicators=0.0,
                technical_depth=0.0,
                communication_clarity=0.0,
                gaps=["Answer too short for meaningful analysis"],
                strengths=[]
            )
        
        # Use semantic analysis if available, otherwise fall back
        if not self.fallback_mode and self.model:
            return self._semantic_analysis(answer, question_category, target_role_level)
        else:
            return self._fallback_analysis(answer, question_category, target_role_level)
    
    def _semantic_analysis(self, answer: str, category: str, role_level: str) -> SemanticScore:
        """Advanced semantic analysis using sentence transformers"""
        
        try:
            # Encode the user's answer
            answer_vector = self.model.encode([answer])[0]
            
            # Get theme vectors for the category
            category_themes = self.theme_vectors.get(category, {})
            
            # Calculate similarities to different themes
            theme_similarities = {}
            for theme, ideal_vector in category_themes.items():
                similarity = np.dot(answer_vector, ideal_vector) / (
                    np.linalg.norm(answer_vector) * np.linalg.norm(ideal_vector)
                )
                theme_similarities[theme] = similarity
            
            # Analyze content coverage
            content_coverage = np.mean(list(theme_similarities.values())) if theme_similarities else 0.0
            
            # Calculate specific indicators
            leadership_score = theme_similarities.get('leadership', 0.0)
            problem_solving_score = theme_similarities.get('problem_solving', 0.0)
            
            # Analyze technical depth based on technical keywords and complexity
            technical_depth = self._calculate_technical_depth(answer, category)
            
            # Analyze communication clarity
            communication_clarity = self._calculate_communication_clarity(answer)
            
            # Calculate depth score based on specificity and detail
            depth_score = self._calculate_depth_score(answer, role_level)
            
            # Calculate relevance score
            relevance_score = max(theme_similarities.values()) if theme_similarities else 0.0
            
            # Identify gaps and strengths
            gaps, strengths = self._identify_gaps_and_strengths(
                theme_similarities, answer, category, role_level
            )
            
            return SemanticScore(
                content_coverage=min(content_coverage, 1.0),
                depth_score=min(depth_score, 1.0),
                relevance_score=min(relevance_score, 1.0),
                leadership_indicators=min(leadership_score, 1.0),
                problem_solving_indicators=min(problem_solving_score, 1.0),
                technical_depth=min(technical_depth, 1.0),
                communication_clarity=min(communication_clarity, 1.0),
                gaps=gaps,
                strengths=strengths
            )
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return self._fallback_analysis(answer, category, role_level)
    
    def _fallback_analysis(self, answer: str, category: str, role_level: str) -> SemanticScore:
        """Fallback analysis using keyword matching and heuristics"""
        
        answer_lower = answer.lower()
        word_count = len(answer.split())
        
        # Leadership indicators
        leadership_keywords = [
            'led', 'managed', 'coordinated', 'organized', 'directed', 'supervised',
            'mentored', 'guided', 'facilitated', 'motivated', 'delegated'
        ]
        leadership_score = min(sum(1 for kw in leadership_keywords if kw in answer_lower) / 5.0, 1.0)
        
        # Problem-solving indicators
        problem_keywords = [
            'analyzed', 'identified', 'evaluated', 'solved', 'resolved', 'investigated',
            'researched', 'diagnosed', 'troubleshot', 'optimized', 'improved'
        ]
        problem_solving_score = min(sum(1 for kw in problem_keywords if kw in answer_lower) / 5.0, 1.0)
        
        # Technical depth
        technical_depth = self._calculate_technical_depth(answer, category)
        
        # Communication clarity
        communication_clarity = self._calculate_communication_clarity(answer)
        
        # Depth score based on length and specificity
        depth_score = self._calculate_depth_score(answer, role_level)
        
        # Content coverage heuristic
        expected_elements = self._get_expected_elements(category)
        covered_elements = sum(1 for elem in expected_elements if elem in answer_lower)
        content_coverage = min(covered_elements / len(expected_elements), 1.0) if expected_elements else 0.5
        
        # Relevance based on category-specific keywords
        relevance_keywords = self._get_relevance_keywords(category)
        relevance_matches = sum(1 for kw in relevance_keywords if kw in answer_lower)
        relevance_score = min(relevance_matches / max(len(relevance_keywords), 1), 1.0)
        
        # Identify gaps and strengths
        gaps, strengths = self._identify_gaps_and_strengths_fallback(
            answer, category, role_level, {
                'leadership': leadership_score,
                'problem_solving': problem_solving_score,
                'technical': technical_depth,
                'communication': communication_clarity
            }
        )
        
        return SemanticScore(
            content_coverage=content_coverage,
            depth_score=depth_score,
            relevance_score=relevance_score,
            leadership_indicators=leadership_score,
            problem_solving_indicators=problem_solving_score,
            technical_depth=technical_depth,
            communication_clarity=communication_clarity,
            gaps=gaps,
            strengths=strengths
        )
    
    def _calculate_technical_depth(self, answer: str, category: str) -> float:
        """Calculate technical depth score"""
        answer_lower = answer.lower()
        
        # Technical terms and concepts
        technical_terms = [
            'api', 'database', 'sql', 'python', 'javascript', 'react', 'node',
            'aws', 'docker', 'kubernetes', 'microservices', 'architecture',
            'scalability', 'performance', 'security', 'testing', 'ci/cd',
            'algorithm', 'data structure', 'framework', 'library', 'protocol'
        ]
        
        # Methodologies and practices
        methodology_terms = [
            'agile', 'scrum', 'devops', 'tdd', 'bdd', 'code review',
            'refactoring', 'optimization', 'debugging', 'monitoring'
        ]
        
        # Weight technical questions higher
        weight = 1.5 if category == 'technical' else 1.0
        
        tech_count = sum(1 for term in technical_terms if term in answer_lower)
        method_count = sum(1 for term in methodology_terms if term in answer_lower)
        
        # Normalize based on answer length
        word_count = len(answer.split())
        normalized_score = ((tech_count + method_count) / max(word_count / 50, 1)) * weight
        
        return min(normalized_score, 1.0)
    
    def _calculate_communication_clarity(self, answer: str) -> float:
        """Calculate communication clarity score"""
        sentences = answer.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        word_count = len(answer.split())
        
        # Ideal sentence length range
        if sentence_count == 0:
            return 0.0
        
        avg_sentence_length = word_count / sentence_count
        
        # Score based on sentence structure
        clarity_score = 0.0
        
        # Good sentence length (10-25 words)
        if 10 <= avg_sentence_length <= 25:
            clarity_score += 0.4
        elif 8 <= avg_sentence_length <= 30:
            clarity_score += 0.2
        
        # Presence of transition words
        transitions = ['first', 'then', 'next', 'finally', 'however', 'therefore', 'because', 'as a result']
        if any(trans in answer.lower() for trans in transitions):
            clarity_score += 0.3
        
        # Specific examples or numbers
        if any(char.isdigit() for char in answer):
            clarity_score += 0.2
        
        # Proper structure indicators
        structure_words = ['situation', 'challenge', 'action', 'result', 'example', 'specifically']
        if any(word in answer.lower() for word in structure_words):
            clarity_score += 0.1
        
        return min(clarity_score, 1.0)
    
    def _calculate_depth_score(self, answer: str, role_level: str) -> float:
        """Calculate depth and specificity score"""
        word_count = len(answer.split())
        
        # Expected word count ranges by role level
        expected_ranges = {
            'junior': (50, 120),
            'mid': (80, 180),
            'senior': (100, 250),
            'lead': (120, 300)
        }
        
        min_words, max_words = expected_ranges.get(role_level, (80, 180))
        
        # Length score
        if min_words <= word_count <= max_words:
            length_score = 1.0
        elif word_count < min_words:
            length_score = word_count / min_words
        else:
            length_score = max(0.7, 1.0 - (word_count - max_words) / max_words)
        
        # Specificity indicators
        specificity_indicators = [
            r'\d+%',  # percentages
            r'\d+ (people|users|clients|team|members)',  # team sizes
            r'\d+ (days|weeks|months|hours)',  # timeframes
            r'(increased|decreased|improved|reduced) by',  # impact metrics
            r'(saved|generated|processed) \$?\d+',  # financial impact
        ]
        
        import re
        specificity_score = 0.0
        for pattern in specificity_indicators:
            if re.search(pattern, answer.lower()):
                specificity_score += 0.2
        
        specificity_score = min(specificity_score, 1.0)
        
        # Combine scores
        return (length_score * 0.6) + (specificity_score * 0.4)
    
    def _get_expected_elements(self, category: str) -> List[str]:
        """Get expected elements for different question categories"""
        elements = {
            'behavioral': ['situation', 'task', 'action', 'result', 'challenge', 'outcome'],
            'technical': ['problem', 'approach', 'implementation', 'testing', 'result', 'trade-off'],
            'situational': ['understanding', 'options', 'decision', 'reasoning', 'follow-up'],
            'motivational': ['passion', 'alignment', 'goals', 'values', 'growth']
        }
        return elements.get(category, [])
    
    def _get_relevance_keywords(self, category: str) -> List[str]:
        """Get relevance keywords for different categories"""
        keywords = {
            'behavioral': ['experience', 'project', 'team', 'challenge', 'leadership', 'collaboration'],
            'technical': ['system', 'code', 'design', 'implementation', 'technology', 'solution'],
            'situational': ['would', 'approach', 'handle', 'decision', 'priority', 'strategy'],
            'motivational': ['passionate', 'interested', 'motivated', 'goal', 'value', 'inspire']
        }
        return keywords.get(category, [])
    
    def _identify_gaps_and_strengths(self, theme_similarities: Dict[str, float], 
                                   answer: str, category: str, role_level: str) -> Tuple[List[str], List[str]]:
        """Identify specific gaps and strengths based on semantic analysis"""
        gaps = []
        strengths = []
        
        # Threshold based on role level
        thresholds = {
            'junior': 0.4,
            'mid': 0.5,
            'senior': 0.6,
            'lead': 0.7
        }
        threshold = thresholds.get(role_level, 0.5)
        
        for theme, score in theme_similarities.items():
            if score >= threshold:
                strengths.append(f"Strong {theme.replace('_', ' ')} demonstration")
            else:
                gaps.append(f"Could strengthen {theme.replace('_', ' ')} examples")
        
        # Add specific content gaps
        if 'result' not in answer.lower() and 'outcome' not in answer.lower():
            gaps.append("Missing specific outcomes or results")
        
        if not any(char.isdigit() for char in answer):
            gaps.append("Add quantifiable metrics or timeframes")
        
        # Add role-level specific gaps
        if role_level in ['senior', 'lead']:
            if 'mentored' not in answer.lower() and 'led' not in answer.lower():
                gaps.append("Senior roles expect leadership/mentoring examples")
        
        return gaps[:3], strengths[:3]  # Limit to top 3 each
    
    def _identify_gaps_and_strengths_fallback(self, answer: str, category: str, 
                                            role_level: str, scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Fallback method for identifying gaps and strengths"""
        gaps = []
        strengths = []
        
        threshold = 0.5
        
        for aspect, score in scores.items():
            if score >= threshold:
                strengths.append(f"Good {aspect.replace('_', ' ')} indicators")
            elif score < 0.3:
                gaps.append(f"Strengthen {aspect.replace('_', ' ')} elements")
        
        # Basic content analysis
        if len(answer.split()) < 50:
            gaps.append("Provide more detailed examples")
        
        if 'I' not in answer:
            gaps.append("Use more personal ownership language")
        
        return gaps[:3], strengths[:3]
    
    def generate_contextualized_feedback(self, semantic_score: SemanticScore, 
                                       category: str, role_level: str, 
                                       word_limit: int = 200) -> str:
        """Generate contextualized feedback within word limits"""
        
        feedback_parts = []
        
        # Overall assessment (30-40 words)
        if semantic_score.content_coverage >= 0.7:
            feedback_parts.append("Strong content coverage with good thematic alignment.")
        elif semantic_score.content_coverage >= 0.5:
            feedback_parts.append("Solid foundation, but room to strengthen key themes.")
        else:
            feedback_parts.append("Content needs significant strengthening to meet expectations.")
        
        # Top strengths (40-50 words)
        if semantic_score.strengths:
            strength_text = " ".join(semantic_score.strengths[:2])
            feedback_parts.append(f"Strengths: {strength_text}.")
        
        # Critical gaps (60-70 words)
        if semantic_score.gaps:
            gap_text = " ".join(semantic_score.gaps[:2])
            feedback_parts.append(f"Key improvements: {gap_text}.")
        
        # Specific scores and recommendations (40-50 words)
        score_feedback = []
        if semantic_score.leadership_indicators >= 0.6:
            score_feedback.append("strong leadership presence")
        elif role_level in ['senior', 'lead']:
            score_feedback.append("add more leadership examples")
        
        if semantic_score.technical_depth >= 0.6:
            score_feedback.append("good technical depth")
        elif category == 'technical':
            score_feedback.append("include more technical specifics")
        
        if score_feedback:
            feedback_parts.append(f"Notable: {', '.join(score_feedback)}.")
        
        # Combine and ensure word limit
        full_feedback = " ".join(feedback_parts)
        words = full_feedback.split()
        
        if len(words) > word_limit:
            # Truncate while keeping complete sentences
            truncated = []
            word_count = 0
            for part in feedback_parts:
                part_words = part.split()
                if word_count + len(part_words) <= word_limit:
                    truncated.append(part)
                    word_count += len(part_words)
                else:
                    break
            full_feedback = " ".join(truncated)
        
        return full_feedback


# Integration function for existing Django views
def integrate_with_existing_views(vectorizer: ContentVectorizer):
    """
    Integration function to enhance the existing analyze_answer_content function
    This would be added to your views.py file
    """
    
    def enhanced_analyze_answer_content(answer: str, question: str, category: str = "behavioral", 
                                      role_level: str = "mid") -> Dict:
        """Enhanced version of your existing analyze_answer_content function"""
        
        # Get your existing analysis
        from views import analyze_answer_content  # Your existing function
        basic_analysis = analyze_answer_content(answer, question)
        
        # Add semantic analysis
        semantic_score = vectorizer.analyze_answer_semantics(answer, category, role_level)
        
        # Generate contextualized feedback
        semantic_feedback = vectorizer.generate_contextualized_feedback(
            semantic_score, category, role_level, word_limit=180
        )
        
        # Combine analyses
        enhanced_analysis = {
            **basic_analysis,  # Your existing analysis
            'semantic_score': semantic_score,
            'semantic_feedback': semantic_feedback,
            'content_coverage': semantic_score.content_coverage,
            'depth_score': semantic_score.depth_score,
            'relevance_score': semantic_score.relevance_score,
            'leadership_indicators': semantic_score.leadership_indicators,
            'problem_solving_indicators': semantic_score.problem_solving_indicators,
            'technical_depth': semantic_score.technical_depth,
            'communication_clarity': semantic_score.communication_clarity
        }
        
        return enhanced_analysis
    
    return enhanced_analyze_answer_content


# Usage example for integration with your Django app
if __name__ == "__main__":
    # Initialize the vectorizer
    vectorizer = ContentVectorizer()
    
    # Example usage
    sample_answer = """
    When I was leading the development team at my previous company, we faced a critical 
    performance issue that was affecting 50,000+ users. I analyzed the system architecture 
    and identified that our database queries were inefficient. I coordinated with the 
    database team to optimize the indexes and refactored the caching layer. As a result, 
    we reduced response times by 70% and improved user satisfaction scores from 3.2 to 4.6.
    """
    
    # Analyze the answer
    semantic_score = vectorizer.analyze_answer_semantics(
        sample_answer, 
        question_category="behavioral", 
        target_role_level="senior"
    )
    
    # Generate feedback
    feedback = vectorizer.generate_contextualized_feedback(
        semantic_score, 
        category="behavioral", 
        role_level="senior",
        word_limit=200
    )
    
    print("Semantic Analysis Results:")
    print(f"Content Coverage: {semantic_score.content_coverage:.2f}")
    print(f"Leadership Indicators: {semantic_score.leadership_indicators:.2f}")
    print(f"Technical Depth: {semantic_score.technical_depth:.2f}")
    print(f"\nFeedback:\n{feedback}")