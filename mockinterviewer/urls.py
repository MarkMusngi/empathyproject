"""
URL configuration for mockinterviewer project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from interview import views  # or whatever your app is actually named


urlpatterns = [
    path('', views.start_page, name='start_page'),
    path('interview/', views.interview_page_with_role_selection, name='interview_page'),
    path('transcribe/', views.transcribe_audio, name='transcribe_audio'),
    path('feedback/', views.feedback_page, name='feedback_page'),
    # Emotion validation endpoints
    path('update-answer-text/', views.update_answer_text, name='update_answer_text'),
    path('emotion-status/', views.get_emotion_status, name='emotion_status'),
    path('correct-emotion/', views.correct_emotion, name='correct_emotion'),
]
