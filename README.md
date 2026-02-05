# Empathic Mock Interview System

A web-based application that simulates mock interviews and provides AI-generated empathetic feedback on user responses. The system integrates Large Language Models (LLMs) with speech-to-text and semantic analysis to evaluate interview answers and return structured performance metrics and supportive feedback.

## Key Features
* **Interactive Interview Interface**: A clean, web-based hub where users receive mock interview questions and record their responses in real-time.
* **Smart Transcription**: Integrated speech-to-text processing that captures verbal answers and converts them into text for analysis.
* **Comprehensive Evaluation**: Manage every detail of an interview assessment including:
    * **Semantic Analysis**: Deep processing of the transcribed response to evaluate content quality and relevance.
    * **Sentiment Tracking**: Analysis of emotion and tone to provide a holistic view of the user's delivery.
    * **Performance Metrics**: Generation of structured ratings and scores based on interview best practices.
* **AI-Powered Feedback**: Utilizes a Large Language Model (Ollama) to generate supportive and empathetic critiques, moving beyond generic automated responses.
* **Modular Feedback Dashboard**: Displays results on a dedicated page featuring emotion metrics, response quality, and actionable advice for improvement.

## Tech Stack
* **Backend**: Python, Django
* **AI Engine**: Large Language Models (Ollama)
* **Processing**: Speech-to-Text and Natural Language Processing (NLP)
* **Frontend**: HTML, CSS, JavaScript
* **API Architecture**: RESTful APIs for seamless data flow between the web interface and the AI model.

## How It Works
1. **Question Presentation**: The system presents a targeted mock interview question to the user through the web interface.
2. **Real-time Capture**: The user responds verbally, and the system performs real-time speech-to-text transcription.
3. **Sentiment & Semantic Analysis**: The transcribed text is processed to identify underlying emotions and the semantic depth of the answer.
4. **LLM Evaluation**: The system sends the processed data to the Ollama model for professional-grade evaluation.
5. **Feedback Generation**: The model creates structured ratings and empathetic, constructive feedback tailored to the user's specific response.
6. **Result Visualization**: Performance metrics and feedback are displayed on a summary dashboard, allowing users to track their progress over time.
