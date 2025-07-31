# Create this as a new file: feedback_models.py


import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class FeedbackContext:
    """Context information for generating personalized feedback"""
    user_profile: Any  # UserProfile instance
    session_data: Dict[str, Any]
    performance_scores: Dict[str, float]
    emotion_data: Dict[str, Any]
    content_analysis: Dict[str, Any]


class BaseFeedbackModel:
    """Base class for all feedback models"""
   
    def __init__(self, name: str, description: str, tone: str = "professional"):
        self.name = name
        self.description = description
        self.tone = tone
        self.usage_count = 0
   
    def generate_feedback(self, context: FeedbackContext) -> str:
        """Generate feedback based on context - to be implemented by subclasses"""
        raise NotImplementedError
   
    def get_opening(self, context: FeedbackContext) -> str:
        """Generate opening section"""
        return "Here's your interview feedback:"
   
    def get_strengths(self, context: FeedbackContext) -> str:
        """Generate strengths section"""
        return "You demonstrated several positive qualities."
   
    def get_improvements(self, context: FeedbackContext) -> str:
        """Generate improvements section"""
        return "Here are areas for development."
   
    def get_emotion_feedback(self, context: FeedbackContext) -> str:
        """Generate emotion-specific feedback"""
        return "Your delivery showed professionalism."
   
    def get_closing(self, context: FeedbackContext) -> str:
        """Generate closing section"""
        return "Keep practicing to continue improving!"


class CoachingModel(BaseFeedbackModel):
    """Supportive, developmental coaching style feedback"""
   
    def __init__(self):
        super().__init__(
            name="coaching",
            description="Supportive and developmental feedback focused on growth",
            tone="supportive"
        )
   
    def generate_feedback(self, context: FeedbackContext) -> str:
        profile = context.user_profile
        scores = context.performance_scores
        emotion = context.emotion_data.get('validated_emotion', 'neutral')
       
        feedback_parts = []
       
        # Encouraging opening
        feedback_parts.append(self.get_opening(context))
       
        # Highlight progress and effort
        if profile.session_count > 1:
            feedback_parts.append(f"🌟 **Progress Recognition**: This is your {profile.session_count}th practice session - your commitment to improvement is excellent!")
       
        # Strengths with encouragement
        feedback_parts.append(self.get_strengths(context))
       
        # Growth-oriented improvements
        feedback_parts.append(self.get_improvements(context))
       
        # Supportive emotion feedback
        feedback_parts.append(self.get_emotion_feedback(context))
       
        # Motivational closing
        feedback_parts.append(self.get_closing(context))
       
        return "\n\n".join(feedback_parts)
   
    def get_opening(self, context: FeedbackContext) -> str:
        name = context.user_profile.user.first_name or "there"
        overall_score = context.performance_scores.get('overall_score', 0.6)
       
        if overall_score >= 0.8:
            return f"👋 Hi {name}! What an impressive response! You're really hitting your stride with these interviews."
        elif overall_score >= 0.6:
            return f"👋 Hi {name}! Great work on this response. You're building solid interview skills and it shows."
        else:
            return f"👋 Hi {name}! Thanks for practicing - every session is a step forward in your interview journey."
   
    def get_strengths(self, context: FeedbackContext) -> str:
        strengths = []
        scores = context.performance_scores
        analysis = context.content_analysis
       
        if scores.get('content_score', 0) >= 0.7:
            strengths.append("your content was well-structured and relevant")
        if scores.get('emotion_score', 0) >= 0.7:
            strengths.append("your delivery showed confidence and engagement")
        if analysis.get('context_provided', False):
            strengths.append("you provided excellent context and specific examples")
        if analysis.get('quantifiable_results'):
            strengths.append("you included impressive quantifiable achievements")
        if analysis.get('learning_mentioned', False):
            strengths.append("you demonstrated growth mindset and learning from experience")
       
        if strengths:
            return f"✨ **What You Did Well**: I noticed {', '.join(strengths)}. These are exactly the qualities interviewers look for!"
        else:
            return "✨ **What You Did Well**: You tackled a challenging question and provided a complete response. That takes courage and shows you're committed to improving."
   
    def get_improvements(self, context: FeedbackContext) -> str:
        improvements = []
        scores = context.performance_scores
        analysis = context.content_analysis
        profile = context.user_profile
       
        if scores.get('content_score', 0) < 0.5:
            improvements.append("try adding more specific details and examples to strengthen your stories")
        if not analysis.get('context_provided', False):
            improvements.append("consider using the STAR method (Situation, Task, Action, Result) to structure your responses")
        if not analysis.get('quantifiable_results'):
            improvements.append("look for opportunities to include numbers, percentages, or measurable outcomes")
        if len(analysis.get('action_words', [])) < 2:
            improvements.append("use more strong action verbs to highlight your personal contributions")
       
        if improvements:
            growth_phrase = "Here's how you can grow even stronger"
            if profile.experience_level in ['entry', 'junior']:
                growth_phrase = "Here are some techniques that will help you shine in interviews"
           
            return f"🎯 **{growth_phrase}**: {', '.join(improvements[:3])}. Remember, these skills develop with practice!"
        else:
            return "🎯 **Keep Building**: You're on the right track! Continue practicing with different question types to build even more confidence."
   
    def get_emotion_feedback(self, context: FeedbackContext) -> str:
        emotion = context.emotion_data.get('validated_emotion', 'neutral')
        confidence = context.emotion_data.get('confidence', 0.5)
       
        emotion_messages = {
            'neutral': "Your calm, professional demeanor is perfect for interviews. This composure will serve you well!",
            'happy': "Your positive energy really came through! This enthusiasm is exactly what employers want to see.",
            'fear': "I can sense some interview nerves - that's completely normal and shows you care! With more practice, your natural confidence will shine through.",
            'angry': "You seemed quite focused and intense. Channel that passion positively - it shows you care deeply about your work.",
            'sad': "You appeared thoughtful and reflective. Remember to let your energy and enthusiasm show through as well.",
            'surprise': "You seemed engaged and reactive to the question. Take your time to collect thoughts - it's perfectly okay to pause and think."
        }
       
        base_message = emotion_messages.get(emotion, "Your delivery showed professionalism.")
       
        if confidence < 0.5:
            base_message += " (Note: The emotion detection had lower confidence, so trust your own sense of how you felt during the response.)"
       
        return f"😊 **Your Delivery**: {base_message}"
   
    def get_closing(self, context: FeedbackContext) -> str:
        profile = context.user_profile
        target_role = profile.target_role or "your dream role"
       
        motivational_endings = [
            f"You're building the skills to land {target_role}. Keep up the excellent work!",
            f"Every practice session brings you closer to {target_role}. I believe in your potential!",
            f"You have what it takes for {target_role}. Trust the process and keep practicing!",
        ]
       
        import random
        return f"🚀 **Keep Going**: {random.choice(motivational_endings)}"


class PeerReviewModel(BaseFeedbackModel):
    """Collaborative, balanced peer review style feedback"""
   
    def __init__(self):
        super().__init__(
            name="peer_review",
            description="Collaborative and balanced feedback as if from a peer",
            tone="collaborative"
        )
   
    def generate_feedback(self, context: FeedbackContext) -> str:
        feedback_parts = []
       
        feedback_parts.append("**Peer Review Feedback**")
        feedback_parts.append(self.get_opening(context))
        feedback_parts.append(self.get_balanced_analysis(context))
        feedback_parts.append(self.get_suggestions(context))
        feedback_parts.append(self.get_emotion_feedback(context))
        feedback_parts.append(self.get_closing(context))
       
        return "\n\n".join(feedback_parts)
   
    def get_opening(self, context: FeedbackContext) -> str:
        return "Hey! I just reviewed your interview response, and here's what I observed as someone who's been through similar interviews:"
   
    def get_balanced_analysis(self, context: FeedbackContext) -> str:
        scores = context.performance_scores
        analysis = context.content_analysis
       
        positives = []
        areas = []
       
        # Balanced assessment
        if scores.get('content_score', 0) >= 0.6:
            positives.append("solid content structure")
        else:
            areas.append("content organization")
       
        if analysis.get('context_provided', False):
            positives.append("good storytelling with context")
        else:
            areas.append("more specific examples")
       
        if analysis.get('quantifiable_results'):
            positives.append("great use of numbers and metrics")
        else:
            areas.append("quantifiable outcomes")
       
        feedback = "📊 **Balanced Take**: "
        if positives:
            feedback += f"You nailed: {', '.join(positives)}. "
        if areas:
            feedback += f"Could strengthen: {', '.join(areas)}."
       
        return feedback
   
    def get_suggestions(self, context: FeedbackContext) -> str:
        profile = context.user_profile
        suggestions = []
       
        # Peer-style suggestions
        if profile.experience_level in ['entry', 'junior']:
            suggestions.append("Try practicing the STAR method - it really helps structure responses")
       
        suggestions.append("Consider recording yourself to catch delivery patterns")
        suggestions.append("Practice with different question types to build versatility")
       
        return f"💡 **What's Helped Me**: {', '.join(suggestions[:2])}. We're all learning together!"
   
    def get_emotion_feedback(self, context: FeedbackContext) -> str:
        emotion = context.emotion_data.get('validated_emotion', 'neutral')
       
        peer_observations = {
            'neutral': "You came across as composed and professional - that's exactly the vibe you want.",
            'happy': "Your enthusiasm really showed! That positive energy is infectious.",
            'fear': "I could sense some nerves, which is totally normal. We've all been there!",
            'angry': "You seemed pretty intense. Make sure to balance that passion with approachability.",
            'sad': "You appeared quite serious. Don't be afraid to show some personality and energy.",
            'surprise': "You looked engaged with the question. Taking time to think is totally fine."
        }
       
        return f"🎭 **Delivery Observation**: {peer_observations.get(emotion, 'Your delivery was professional.')}"
   
    def get_closing(self, context: FeedbackContext) -> str:
        return "🤝 **Final Thought**: You're doing great by practicing! Feel free to reach out if you want to do mock interviews together. We've got this!"


class ExpertAnalysisModel(BaseFeedbackModel):
    """Technical, detailed expert analysis style feedback"""
   
    def __init__(self):
        super().__init__(
            name="expert_analysis",
            description="Technical and detailed analysis with specific recommendations",
            tone="analytical"
        )
   
    def generate_feedback(self, context: FeedbackContext) -> str:
        feedback_parts = []
       
        feedback_parts.append("**Expert Analysis Report**")
        feedback_parts.append(self.get_performance_breakdown(context))
        feedback_parts.append(self.get_technical_analysis(context))
        feedback_parts.append(self.get_strategic_recommendations(context))
        feedback_parts.append(self.get_emotion_analysis(context))
        feedback_parts.append(self.get_benchmarking(context))
       
        return "\n\n".join(feedback_parts)
   
    def get_performance_breakdown(self, context: FeedbackContext) -> str:
        scores = context.performance_scores
       
        breakdown = "📈 **Performance Metrics**:\n"
        breakdown += f"• Overall Score: {scores.get('overall_score', 0.6):.1%}\n"
        breakdown += f"• Content Quality: {scores.get('content_score', 0.6):.1%}\n"
        breakdown += f"• Delivery Confidence: {scores.get('emotion_score', 0.5):.1%}\n"
       
        semantic_analysis = context.content_analysis.get('semantic_analysis')
        if semantic_analysis:
            breakdown += f"• Semantic Relevance: {semantic_analysis.get('relevance_score', 0.6):.1%}\n"
            breakdown += f"• Technical Depth: {semantic_analysis.get('technical_depth', 0.5):.1%}\n"
            breakdown += f"• Leadership Indicators: {semantic_analysis.get('leadership_indicators', 0.5):.1%}"
       
        return breakdown
   
    def get_technical_analysis(self, context: FeedbackContext) -> str:
        analysis = context.content_analysis
       
        technical_findings = []
       
        # Content structure analysis
        if analysis.get('context_provided', False):
            technical_findings.append("✓ Contextual framework established")
        else:
            technical_findings.append("⚠ Missing situational context")
       
        # Action analysis
        action_count = len(analysis.get('action_words', []))
        if action_count >= 3:
            technical_findings.append(f"✓ Strong action orientation ({action_count} action verbs)")
        else:
            technical_findings.append(f"⚠ Limited action vocabulary ({action_count} action verbs)")
       
        # Results analysis
        if analysis.get('quantifiable_results'):
            technical_findings.append(f"✓ Quantified outcomes ({len(analysis['quantifiable_results'])} metrics)")
        else:
            technical_findings.append("⚠ No quantifiable results provided")
       
        return f"🔍 **Technical Analysis**:\n" + "\n".join(f"  {finding}" for finding in technical_findings)
   
    def get_strategic_recommendations(self, context: FeedbackContext) -> str:
        profile = context.user_profile
        analysis = context.content_analysis
       
        recommendations = []
       
        # Experience-level specific recommendations
        if profile.experience_level in ['senior', 'lead', 'executive']:
            recommendations.append("Emphasize strategic thinking and leadership impact")
            recommendations.append("Include organizational influence and system-level changes")
        elif profile.experience_level in ['mid']:
            recommendations.append("Balance individual contributions with team collaboration")
            recommendations.append("Demonstrate progression from execution to ownership")
        else:
            recommendations.append("Focus on learning agility and growth potential")
            recommendations.append("Highlight specific technical or functional skills gained")
       
        # Role-type specific recommendations
        if profile.role_type == 'technical':
            recommendations.append("Include technical architecture and implementation details")
        elif profile.role_type == 'management':
            recommendations.append("Emphasize people leadership and organizational outcomes")
       
        return f"🎯 **Strategic Recommendations**:\n" + "\n".join(f"  • {rec}" for rec in recommendations[:4])
   
    def get_emotion_analysis(self, context: FeedbackContext) -> str:
        emotion_data = context.emotion_data
        emotion = emotion_data.get('validated_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.5)
        stability = emotion_data.get('stability_score', 0.5)
       
        analysis = f"🧠 **Behavioral Analysis**:\n"
        analysis += f"  • Detected State: {emotion.title()} (confidence: {confidence:.1%})\n"
        analysis += f"  • Emotional Stability: {stability:.1%}\n"
       
        # Professional interpretation
        if emotion == 'neutral' and confidence > 0.7:
            analysis += "  • Assessment: Optimal professional composure maintained"
        elif emotion == 'happy' and confidence > 0.6:
            analysis += "  • Assessment: Positive engagement detected - excellent for cultural fit"
        elif emotion in ['fear', 'sad'] and confidence > 0.6:
            analysis += "  • Assessment: Stress indicators present - recommend confidence building"
        else:
            analysis += "  • Assessment: Variable emotional state - consider preparation strategies"
       
        return analysis
   
    def get_benchmarking(self, context: FeedbackContext) -> str:
        profile = context.user_profile
        scores = context.performance_scores
        overall_score = scores.get('overall_score', 0.6)
       
        # Benchmarking based on experience level
        benchmarks = {
            'entry': {'good': 0.6, 'excellent': 0.75},
            'junior': {'good': 0.65, 'excellent': 0.8},
            'mid': {'good': 0.7, 'excellent': 0.85},
            'senior': {'good': 0.75, 'excellent': 0.9},
            'lead': {'good': 0.8, 'excellent': 0.92},
            'executive': {'good': 0.85, 'excellent': 0.95}
        }
       
        level_benchmarks = benchmarks.get(profile.experience_level, benchmarks['mid'])
       
        benchmark_text = f"📊 **Performance Benchmarking**:\n"
        benchmark_text += f"  • Your Score: {overall_score:.1%}\n"
        benchmark_text += f"  • {profile.get_experience_level_display()} Benchmark - Good: {level_benchmarks['good']:.1%}, Excellent: {level_benchmarks['excellent']:.1%}\n"
       
        if overall_score >= level_benchmarks['excellent']:
            benchmark_text += "  • **Status**: Exceeds expectations for experience level"
        elif overall_score >= level_benchmarks['good']:
            benchmark_text += "  • **Status**: Meets expectations for experience level"
        else:
            benchmark_text += "  • **Status**: Below benchmark - focus on fundamental improvements"
       
        return benchmark_text


class StrategicModel(BaseFeedbackModel):
    """Big picture, leadership-focused strategic feedback"""
   
    def __init__(self):
        super().__init__(
            name="strategic",
            description="Big picture and leadership-focused feedback",
            tone="executive"
        )
   
    def generate_feedback(self, context: FeedbackContext) -> str:
        feedback_parts = []
       
        feedback_parts.append("**Strategic Leadership Assessment**")
        feedback_parts.append(self.get_strategic_overview(context))
        feedback_parts.append(self.get_leadership_indicators(context))
        feedback_parts.append(self.get_business_impact(context))
        feedback_parts.append(self.get_executive_presence(context))
        feedback_parts.append(self.get_strategic_recommendations(context))
       
        return "\n\n".join(feedback_parts)
   
    def get_strategic_overview(self, context: FeedbackContext) -> str:
        profile = context.user_profile
        scores = context.performance_scores
       
        overview = f"🎯 **Executive Summary**: "
       
        if scores.get('overall_score', 0.6) >= 0.8:
            overview += f"Demonstrated strong alignment with {profile.get_experience_level_display()} leadership expectations. "
        else:
            overview += f"Opportunity to strengthen leadership narrative for {profile.get_experience_level_display()} role. "
       
        overview += f"Response shows potential for {profile.target_role or 'senior leadership'} trajectory."
       
        return overview
   
    def get_leadership_indicators(self, context: FeedbackContext) -> str:
        analysis = context.content_analysis
        semantic_analysis = analysis.get('semantic_analysis', {})
       
        leadership_score = semantic_analysis.get('leadership_indicators', 0.5)
       
        indicators = f"👑 **Leadership Presence**: {leadership_score:.1%}\n"
       
        leadership_elements = []
        if analysis.get('context_provided', False):
            leadership_elements.append("Situational awareness and context-setting")
        if len(analysis.get('action_words', [])) >= 3:
            leadership_elements.append("Action-oriented decision making")
        if analysis.get('quantifiable_results'):
            leadership_elements.append("Results-driven mindset")
        if analysis.get('learning_mentioned', False):
            leadership_elements.append("Growth leadership and adaptability")
       
        if leadership_elements:
            indicators += "  **Demonstrated**: " + ", ".join(leadership_elements)
        else:
            indicators += "  **Recommendation**: Strengthen leadership narrative with specific examples of influence and impact"
       
        return indicators
   
    def get_business_impact(self, context: FeedbackContext) -> str:
        analysis = context.content_analysis
       
        impact_analysis = "💼 **Business Impact Assessment**:\n"
       
        if analysis.get('quantifiable_results'):
            impact_analysis += f"  ✓ Quantified business outcomes: {len(analysis['quantifiable_results'])} metrics provided\n"
        else:
            impact_analysis += "  ⚠ Missing quantified business impact\n"
       
        if analysis.get('challenges_discussed'):
            impact_analysis += "  ✓ Problem-solving orientation demonstrated\n"
        else:
            impact_analysis += "  ⚠ Limited evidence of strategic problem-solving\n"
       
        if len(analysis.get('skills_mentioned', [])) >= 2:
            impact_analysis += "  ✓ Multi-functional capability indicated\n"
        else:
            impact_analysis += "  ⚠ Narrow functional focus\n"
       
        impact_analysis += "\n  **Strategic Recommendation**: Emphasize organizational-level impact and cross-functional influence"
       
        return impact_analysis
   
    def get_executive_presence(self, context: FeedbackContext) -> str:
        emotion_data = context.emotion_data
        emotion = emotion_data.get('validated_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.5)
       
        presence_analysis = "🎭 **Executive Presence**: "
       
        if emotion == 'neutral' and confidence > 0.7:
            presence_analysis += "Strong composure and professional bearing. Demonstrates executive-level emotional regulation."
        elif emotion == 'happy' and confidence > 0.6:
            presence_analysis += "Positive energy and engagement. Consider balancing enthusiasm with gravitas for senior roles."
        elif emotion in ['fear', 'sad']:
            presence_analysis += "Opportunity to strengthen executive presence. Focus on projecting confidence and authority."
        else:
            presence_analysis += "Variable presence indicators. Develop consistent professional demeanor."
       
        return presence_analysis
   
    def get_strategic_recommendations(self, context: FeedbackContext) -> str:
        profile = context.user_profile
       
        recommendations = []
       
        # Strategic recommendations based on target level
        if profile.experience_level in ['senior', 'lead', 'executive']:
            recommendations.extend([
                "Frame responses in terms of organizational strategy and vision",
                "Emphasize stakeholder management and influence without authority",
                "Demonstrate systems thinking and long-term impact consideration"
            ])
        else:
            recommendations.extend([
                "Build strategic thinking examples into responses",
                "Connect individual contributions to broader business objectives",
                "Develop leadership stories that show progression and growth"
            ])
       
        # Industry-specific strategic advice
        if profile.industry == 'technology':
            recommendations.append("Incorporate digital transformation and innovation leadership")
        elif profile.industry == 'finance':
            recommendations.append("Emphasize risk management and regulatory awareness")
        elif profile.industry == 'healthcare':
            recommendations.append("Highlight patient outcomes and regulatory compliance")
       
        return f"🚀 **Strategic Development Path**:\n" + "\n".join(f"  • {rec}" for rec in recommendations[:4])


class AdaptiveModel(BaseFeedbackModel):
    """Adaptive model that selects best approach based on user data"""
   
    def __init__(self):
        super().__init__(
            name="adaptive",
            description="Automatically adjusts feedback style based on user performance and preferences",
            tone="adaptive"
        )
        self.models = {
            'coaching': CoachingModel(),
            'peer_review': PeerReviewModel(),
            'expert_analysis': ExpertAnalysisModel(),
            'strategic': StrategicModel()
        }
   
    def select_best_model(self, context: FeedbackContext) -> BaseFeedbackModel:
        """Select the most appropriate feedback model based on context"""
        profile = context.user_profile
       
        # Get user's preferred model if not adaptive
        if profile.preferred_feedback_model != 'adaptive':
            return self.models.get(profile.preferred_feedback_model, self.models['coaching'])
       
        # Use effectiveness data if available
        recommended_model = profile.get_recommended_feedback_model()
        if recommended_model in self.models:
            return self.models[recommended_model]
       
        # Fallback selection based on profile characteristics
        if profile.session_count <= 3:
            return self.models['coaching']  # New users get supportive feedback
        elif profile.experience_level in ['senior', 'lead', 'executive']:
            return self.models['strategic']  # Senior levels get strategic feedback
        elif profile.learning_preference == 'data_driven':
            return self.models['expert_analysis']  # Data-driven users get detailed analysis
        elif profile.learning_preference == 'conversational':
            return self.models['peer_review']  # Conversational users get peer style
        else:
            return self.models['coaching']  # Default to coaching
   
    def generate_feedback(self, context: FeedbackContext) -> str:
        selected_model = self.select_best_model(context)
       
        # Generate feedback with selected model
        feedback = selected_model.generate_feedback(context)
       
        # Add adaptive note
        model_note = f"\n\n---\n*Feedback Style: {selected_model.description}*"
        if context.user_profile.preferred_feedback_model == 'adaptive':
            model_note += f"\n*Auto-selected based on your profile and performance history.*"
       
        return feedback + model_note


class FeedbackModelManager:
    """Manager class for handling all feedback models"""
   
    def __init__(self):
        self.models = {
            'coaching': CoachingModel(),
            'peer_review': PeerReviewModel(),
            'expert_analysis': ExpertAnalysisModel(),
            'strategic': StrategicModel(),
            'adaptive': AdaptiveModel()
        }
   
    def get_model(self, model_name: str) -> BaseFeedbackModel:
        """Get a specific feedback model"""
        return self.models.get(model_name, self.models['coaching'])
   
    def generate_feedback(self, model_name: str, context: FeedbackContext) -> str:
        """Generate feedback using specified model"""
        model = self.get_model(model_name)
        return model.generate_feedback(context)
   
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models with descriptions"""
        return {name: model.description for name, model in self.models.items()}
   
    def update_model_effectiveness(self, model_name: str, user_profile, user_rating: int, engagement_score: float):
        """Update effectiveness tracking for a model"""
        user_profile.update_feedback_effectiveness(model_name, user_rating, engagement_score)


# Utility function to create feedback context
def create_feedback_context(user_profile, session_data: Dict, performance_scores: Dict,
                          emotion_data: Dict, content_analysis: Dict) -> FeedbackContext:
    """Helper function to create FeedbackContext"""
    return FeedbackContext(
        user_profile=user_profile,
        session_data=session_data,
        performance_scores=performance_scores,
        emotion_data=emotion_data,
        content_analysis=content_analysis
    )

