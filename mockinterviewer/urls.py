
from django.urls import path
from interview import views  # or whatever your app is actually named

urlpatterns = [
    # Main pages
    path('', views.start_page, name='start_page'),
    path('interview/', views.interview_page_with_role_selection, name='interview_page'),
    path('transcribe/', views.transcribe_audio, name='transcribe_audio'),
    path('feedback/', views.feedback_page_enhanced_with_recording, name='feedback_page'),
    path('start/', views.start_page, name='start_page'),
    # Emotion validation endpoints (existing)
    path('update-answer-text/', views.update_answer_text, name='update_answer_text'),
    path('emotion-status/', views.get_emotion_status, name='emotion_status'),
    path('correct-emotion/', views.correct_emotion, name='correct_emotion'),
    path('emotion-metrics/', views.get_emotion_metrics, name='emotion_metrics'),
    path('emotion-confirmation-overlay/', views.emotion_confirmation_overlay, name='emotion_confirmation_overlay'),
    
    # NEW: Enhanced emotion endpoints (required by interview.html)
    path('quick-emotion-correction/', views.quick_emotion_correction, name='quick_emotion_correction'),
    path('emotion-history-timeline/', views.emotion_history_timeline, name='emotion_history_timeline'),
    path('emotion-impact-explanation/', views.emotion_impact_explanation, name='emotion_impact_explanation'),
    path('emotion-detection-accuracy/', views.emotion_detection_accuracy, name='emotion_detection_accuracy'),
    path('reset-emotion-session/', views.reset_emotion_session, name='reset_emotion_session'),
    
    # NEW: Profile management endpoints (required by start.html and profile_setup.html)
    path('create-profile/', views.create_profile, name='create_profile'),
    path('get-profile/', views.get_profile, name='get_profile'),
    path('update-profile-preferences/', views.update_profile_preferences, name='update_profile_preferences'),
    
    # NEW: Feedback and analytics endpoints (required by feedback.html and analytics.html)
    path('rate-feedback/', views.rate_feedback, name='rate_feedback'),
    path('get-feedback-models/', views.get_feedback_models, name='get_feedback_models'),
    path('get-user-analytics/', views.get_user_analytics, name='get_user_analytics'),
    
    # Utility endpoints
    path('set-role-level/', views.set_role_level, name='set_role_level'),
    path('set-emotion/', views.set_emotion, name='set_emotion'),
    
    # System status and diagnostics
    path('vectorization-status/', views.vectorization_status, name='vectorization_status'),
    path('diagnostic-info/', views.diagnostic_info, name='diagnostic_info'),
    #path('analytics/', views.analytics_page, name='analytics_page'),
    path('analytics/', views.analytics_page, name='analytics'),
    path('debug_metrics/', views.debug_metrics, name='debug_metrics'),
    path('get_emotion_metrics/', views.get_emotion_metrics, name='get_emotion_metrics'),
    path('emotion_detection_accuracy/', views.emotion_detection_accuracy, name='emotion_detection_accuracy'),
    # Add these to your urlpatterns:
    path('get-session-analytics/', views.get_session_analytics, name='get_session_analytics'),
    path('record-session-data/', views.record_session_data, name='record_session_data'),
    path('clear-user-analytics/', views.clear_user_analytics, name='clear_user_analytics'),
]