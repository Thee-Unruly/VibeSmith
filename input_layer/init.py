"""
Input layer for VibeSmith - Emotion detection from various sources
"""
from .explicit_input import ExplicitInput
from .text_emotion import TextEmotionAnalyzer
from .speech_emotion import SpeechEmotionRecognizer
from .face_emotion import FacialEmotionDetector

__all__ = [
    'ExplicitInput',
    'TextEmotionAnalyzer', 
    'SpeechEmotionRecognizer',
    'FacialEmotionDetector'
]