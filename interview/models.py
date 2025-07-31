from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
import json


class UserProfile(models.Model):
    """Extended user profile with interview-specific information"""
   
    EXPERIENCE_LEVELS = [
        ('entry', 'Entry Level (0-2 years)'),
        ('junior', 'Junior (2-4 years)'),
        ('mid', 'Mid Level (4-7 years)'),
        ('senior', 'Senior (7-12 years)'),
        ('lead', 'Lead/Manager (12+ years)'),
        ('executive', 'Executive/C-Level'),
    ]
   
    INDUSTRIES = [
        ('technology', 'Technology'),
        ('finance', 'Finance'),
        ('healthcare', 'Healthcare'),
        ('education', 'Education'),
        ('consulting', 'Consulting'),
        ('retail', 'Retail'),
        ('manufacturing', 'Manufacturing'),
        ('government', 'Government'),
        ('nonprofit', 'Non-Profit'),
        ('other', 'Other'),
    ]
   
    ROLE_TYPES = [
        ('technical', 'Technical/Engineering'),
        ('product', 'Product Management'),
        ('sales', 'Sales/Business Development'),
        ('marketing', 'Marketing'),
        ('operations', 'Operations'),
        ('finance', 'Finance/Accounting'),
        ('hr', 'Human Resources'),
        ('design', 'Design/Creative'),
        ('management', 'Management/Leadership'),
        ('other', 'Other'),
    ]
   
    FEEDBACK_MODELS = [
        ('coaching', 'Coaching Style - Supportive and developmental'),
        ('peer_review', 'Peer Review - Collaborative and balanced'),
        ('expert_analysis', 'Expert Analysis - Technical and detailed'),
        ('strategic', 'Strategic Focus - Big picture and leadership'),
        ('adaptive', 'Adaptive - Automatically adjust based on performance'),
    ]
   
    LEARNING_PREFERENCES = [
        ('detailed', 'Detailed - I want comprehensive feedback'),
        ('concise', 'Concise - Give me the key points quickly'),
        ('structured', 'Structured - Use frameworks and templates'),
        ('conversational', 'Conversational - Natural, discussion-style'),
        ('data_driven', 'Data-Driven - Show me metrics and scores'),
    ]
   
    user = models.OneToOneField(User, on_delete=models.CASCADE)
   
    # Profile Information
    experience_level = models.CharField(max_length=20, choices=EXPERIENCE_LEVELS, default='mid')
    industry = models.CharField(max_length=20, choices=INDUSTRIES, default='technology')
    role_type = models.CharField(max_length=20, choices=ROLE_TYPES, default='technical')
    current_job_title = models.CharField(max_length=100, blank=True)
    target_role = models.CharField(max_length=100, blank=True)
   
    # Feedback Preferences
    preferred_feedback_model = models.CharField(max_length=20, choices=FEEDBACK_MODELS, default='adaptive')
    learning_preference = models.CharField(max_length=20, choices=LEARNING_PREFERENCES, default='detailed')
    feedback_complexity = models.IntegerField(default=3, help_text="1=Basic, 5=Advanced")  # 1-5 scale
   
    # Goals and Focus Areas
    interview_goals = models.TextField(blank=True, help_text="What are your interview goals?")
    focus_areas = models.JSONField(default=list, help_text="Areas to focus on (JSON list)")
    weak_areas = models.JSONField(default=list, help_text="Areas needing improvement (JSON list)")
   
    # Personalization Data
    session_count = models.IntegerField(default=0)
    total_practice_time = models.DurationField(default=timezone.timedelta)
    average_confidence = models.FloatField(default=0.0)
    preferred_question_types = models.JSONField(default=list)
   
    # Adaptive Learning
    performance_history = models.JSONField(default=dict, help_text="Historical performance data")
    engagement_metrics = models.JSONField(default=dict, help_text="User engagement patterns")
    feedback_effectiveness = models.JSONField(default=dict, help_text="Feedback model effectiveness scores")
   
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    onboarding_completed = models.BooleanField(default=False)
    profile_completion = models.FloatField(default=0.0)  # 0-1 scale
   
    def __str__(self):
        return f"{self.user.username} - {self.get_experience_level_display()}"
   
    def get_profile_completion(self):
        """Calculate profile completion percentage"""
        fields_to_check = [
            'experience_level', 'industry', 'role_type', 'current_job_title',
            'target_role', 'preferred_feedback_model', 'learning_preference',
            'interview_goals'
        ]
       
        completed_fields = sum(1 for field in fields_to_check if getattr(self, field))
        completion = completed_fields / len(fields_to_check)
       
        # Update stored completion
        self.profile_completion = completion
        self.save(update_fields=['profile_completion'])
       
        return completion
   
    def update_performance_history(self, session_data):
        """Update performance history with new session data"""
        if not self.performance_history:
            self.performance_history = {
                'sessions': [],
                'avg_scores': {},
                'improvement_trend': [],
                'category_performance': {}
            }
       
        # Add new session
        self.performance_history['sessions'].append({
            'timestamp': session_data.get('timestamp', timezone.now().isoformat()),
            'category': session_data.get('category', 'behavioral'),
            'overall_score': session_data.get('overall_score', 0.6),
            'emotion_confidence': session_data.get('emotion_confidence', 0.5),
            'content_score': session_data.get('content_score', 0.6),
            'feedback_model_used': session_data.get('feedback_model', self.preferred_feedback_model),
            'user_rating': session_data.get('user_rating', None)
        })
       
        # Keep only last 50 sessions
        self.performance_history['sessions'] = self.performance_history['sessions'][-50:]
       
        # Update metrics
        self.session_count += 1
        self.save()
   
    def get_recommended_feedback_model(self):
        """Recommend best feedback model based on performance and preferences"""
        if self.preferred_feedback_model != 'adaptive':
            return self.preferred_feedback_model
       
        # Analyze effectiveness of different models
        if not self.feedback_effectiveness:
            # Default recommendation based on experience level
            if self.experience_level in ['entry', 'junior']:
                return 'coaching'
            elif self.experience_level in ['senior', 'lead', 'executive']:
                return 'strategic'
            else:
                return 'expert_analysis'
       
        # Find most effective model
        best_model = max(self.feedback_effectiveness.items(), key=lambda x: x[1].get('effectiveness', 0.5))
        return best_model[0]
   
    def update_feedback_effectiveness(self, model_name, user_rating, engagement_score):
        """Update effectiveness scores for feedback models"""
        if not self.feedback_effectiveness:
            self.feedback_effectiveness = {}
       
        if model_name not in self.feedback_effectiveness:
            self.feedback_effectiveness[model_name] = {
                'usage_count': 0,
                'total_rating': 0,
                'total_engagement': 0,
                'effectiveness': 0.5
            }
       
        model_data = self.feedback_effectiveness[model_name]
        model_data['usage_count'] += 1
        model_data['total_rating'] += user_rating
        model_data['total_engagement'] += engagement_score
       
        # Calculate weighted effectiveness
        avg_rating = model_data['total_rating'] / model_data['usage_count']
        avg_engagement = model_data['total_engagement'] / model_data['usage_count']
        model_data['effectiveness'] = (avg_rating * 0.6) + (avg_engagement * 0.4)
       
        self.save()




class InterviewSession(models.Model):
    """Track individual interview sessions"""
   
    CATEGORIES = [
        ('behavioral', 'Behavioral'),
        ('technical', 'Technical'),
        ('situational', 'Situational'),
        ('motivational', 'Motivational'),
    ]
   
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='sessions')
   
    # Session Details
    category = models.CharField(max_length=20, choices=CATEGORIES)
    question = models.TextField()
    answer = models.TextField()
   
    # Scores and Metrics
    overall_score = models.FloatField(default=0.0)
    content_score = models.FloatField(default=0.0)
    emotion_score = models.FloatField(default=0.0)
    delivery_score = models.FloatField(default=0.0)
   
    # Emotion Data
    detected_emotion = models.CharField(max_length=20, default='neutral')
    validated_emotion = models.CharField(max_length=20, default='neutral')
    emotion_confidence = models.FloatField(default=0.0)
    emotion_corrections = models.IntegerField(default=0)
   
    # Feedback Data
    feedback_model_used = models.CharField(max_length=20, default='adaptive')
    feedback_content = models.TextField()
    feedback_generated_at = models.DateTimeField(auto_now_add=True)
   
    # User Feedback
    user_rating = models.IntegerField(null=True, blank=True, help_text="1-5 rating of feedback quality")
    user_notes = models.TextField(blank=True)
   
    # Technical Metrics
    processing_time = models.FloatField(default=0.0)
    vectorizer_used = models.BooleanField(default=False)
   
    # Session Metadata
    duration = models.DurationField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
   
    class Meta:
        ordering = ['-created_at']
   
    def __str__(self):
        return f"{self.user_profile.user.username} - {self.category} - {self.created_at.strftime('%Y-%m-%d')}"




class FeedbackTemplate(models.Model):
    """Templates for different feedback models"""
   
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField()
   
    # Template Structure
    opening_template = models.TextField(help_text="Opening section template")
    strength_template = models.TextField(help_text="Strengths section template")
    improvement_template = models.TextField(help_text="Improvements section template")
    emotion_template = models.TextField(help_text="Emotion feedback template")
    closing_template = models.TextField(help_text="Closing section template")
   
    # Model Characteristics
    tone = models.CharField(max_length=20, default='professional')  # supportive, direct, analytical, etc.
    complexity_level = models.IntegerField(default=3, help_text="1=Basic, 5=Advanced")
    focus_areas = models.JSONField(default=list, help_text="Primary focus areas for this model")
   
    # Usage Metrics
    usage_count = models.IntegerField(default=0)
    average_rating = models.FloatField(default=0.0)
   
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
   
    def __str__(self):
        return f"{self.name} - {self.tone}"




# Signal to create UserProfile when User is created
from django.db.models.signals import post_save
from django.dispatch import receiver


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'userprofile'):
        instance.userprofile.save()
    else:
        UserProfile.objects.create(user=instance)

