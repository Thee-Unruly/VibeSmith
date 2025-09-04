# NLP emotion classifier
"""
Advanced Text-based emotion detection using NLP techniques
"""
import re
from collections import Counter
import numpy as np

class TextEmotionAnalyzer:
    # Expanded emotion lexicon with weights
    EMOTION_LEXICON = {
        'joy': {
            'happy': 1.0, 'joy': 1.2, 'excited': 1.1, 'delighted': 1.3, 'pleased': 1.0,
            'ecstatic': 1.5, 'thrilled': 1.3, 'bliss': 1.4, 'cheerful': 1.1, 'content': 0.9,
            'jubilant': 1.4, 'elated': 1.3, 'euphoric': 1.5, 'wonderful': 1.0, 'awesome': 1.0,
            'great': 0.9, 'fantastic': 1.1, 'amazing': 1.0, 'love': 1.2, 'loving': 1.1,
            'smile': 0.8, 'laugh': 0.9, 'celebrate': 1.0, 'victory': 0.9, 'win': 0.8
        },
        'sadness': {
            'sad': 1.0, 'unhappy': 1.1, 'depressed': 1.5, 'gloomy': 1.2, 'miserable': 1.4,
            'heartbroken': 1.6, 'sorrow': 1.3, 'melancholy': 1.4, 'disappointed': 1.2,
            'down': 1.0, 'blue': 1.0, 'grief': 1.5, 'tear': 0.9, 'cry': 1.0, 'loss': 1.1,
            'alone': 1.0, 'lonely': 1.2, 'hurt': 1.1, 'pain': 1.0, 'regret': 1.1
        },
        'anger': {
            'angry': 1.0, 'mad': 1.1, 'furious': 1.5, 'outraged': 1.4, 'irate': 1.3,
            'annoyed': 1.0, 'frustrated': 1.2, 'hostile': 1.3, 'irritated': 1.1,
            'enraged': 1.5, 'livid': 1.4, 'incensed': 1.3, 'hate': 1.3, 'hatred': 1.3,
            'rage': 1.4, 'fury': 1.4, 'resent': 1.2, 'bitter': 1.1
        },
        'fear': {
            'scared': 1.0, 'afraid': 1.1, 'fearful': 1.2, 'terrified': 1.5, 'panicked': 1.4,
            'nervous': 1.0, 'anxious': 1.2, 'worried': 1.1, 'apprehensive': 1.3,
            'dread': 1.4, 'horror': 1.3, 'phobia': 1.2, 'panic': 1.3, 'terror': 1.4,
            'uneasy': 1.0, 'threatened': 1.2, 'intimidated': 1.1
        },
        'surprise': {
            'surprised': 1.0, 'shocked': 1.3, 'amazed': 1.2, 'astonished': 1.4,
            'astounded': 1.4, 'stunned': 1.3, 'startled': 1.1, 'unexpected': 1.2,
            'bewildered': 1.2, 'dumbfounded': 1.3, 'flabbergasted': 1.4, 'awe': 1.2
        },
        'trust': {
            'trust': 1.0, 'confident': 1.1, 'secure': 1.0, 'safe': 0.9, 'reliable': 0.9,
            'faith': 1.1, 'believe': 1.0, 'certain': 1.0, 'dependable': 0.9, 'sure': 0.8,
            'conviction': 1.1, 'credible': 0.8, 'trustworthy': 0.9, 'loyal': 0.9
        },
        'disgust': {
            'disgust': 1.0, 'disgusted': 1.1, 'revolted': 1.3, 'repulsed': 1.3,
            'sickened': 1.4, 'nauseated': 1.3, 'appalled': 1.2, 'horrified': 1.3,
            'contempt': 1.2, 'loathe': 1.3, 'abhor': 1.4, 'repulsive': 1.2, 'gross': 1.0
        },
        'anticipation': {
            'anticipate': 1.0, 'expect': 0.9, 'await': 0.9, 'hope': 1.1, 'excited': 1.0,
            'eager': 1.2, 'looking forward': 1.3, 'foresee': 1.0, 'predict': 0.9,
            'prepare': 0.8, 'await': 0.9, 'optimistic': 1.1, 'expectant': 1.0
        }
    }
    
    # Intensifiers and negators
    INTENSIFIERS = {
        'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'quite': 1.2, 'so': 1.4,
        'too': 1.3, 'absolutely': 1.8, 'completely': 1.6, 'totally': 1.5,
        'utterly': 1.7, 'highly': 1.4, 'especially': 1.3
    }
    
    NEGATORS = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere'}
    
    def __init__(self, use_advanced_model=False):
        self.use_advanced_model = use_advanced_model
        
    def analyze_text(self, text):
        """
        Analyze emotion from text input with advanced NLP techniques
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Emotion probability distribution
        """
        if not text or not text.strip():
            return self._neutral_emotion()
            
        # Preprocess text
        sentences = self._split_into_sentences(text)
        emotion_scores = {emotion: 0.0 for emotion in self.EMOTION_LEXICON.keys()}
        
        # Analyze each sentence
        for sentence in sentences:
            sentence_scores = self._analyze_sentence(sentence)
            for emotion, score in sentence_scores.items():
                emotion_scores[emotion] += score
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_probs = {emotion: score/total for emotion, score in emotion_scores.items()}
        else:
            emotion_probs = self._neutral_emotion()
            
        return emotion_probs
    
    def _split_into_sentences(self, text):
        """
        Split text into sentences using regex
        """
        # Simple sentence splitting - can be enhanced with NLP libraries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _analyze_sentence(self, sentence):
        """
        Analyze emotion in a single sentence
        """
        words = re.findall(r'\b\w+\b', sentence.lower())
        emotion_scores = {emotion: 0.0 for emotion in self.EMOTION_LEXICON.keys()}
        
        # Check for emotion words with context
        for i, word in enumerate(words):
            emotion, intensity = self._get_emotion_for_word(word)
            if emotion:
                # Check for intensifiers and negators
                modifier = 1.0
                if i > 0:
                    prev_word = words[i-1]
                    if prev_word in self.INTENSIFIERS:
                        modifier *= self.INTENSIFIERS[prev_word]
                    elif prev_word in self.NEGATORS:
                        modifier *= -1.0  # Negate the emotion
                
                emotion_scores[emotion] += intensity * modifier
        
        return emotion_scores
    
    def _get_emotion_for_word(self, word):
        """
        Find which emotion a word belongs to and its intensity
        """
        for emotion, words_dict in self.EMOTION_LEXICON.items():
            if word in words_dict:
                return emotion, words_dict[word]
        return None, 0
    
    def _neutral_emotion(self):
        """
        Return neutral emotion distribution
        """
        emotions = list(self.EMOTION_LEXICON.keys())
        return {emotion: 1.0/len(emotions) for emotion in emotions}
    
    def get_dominant_emotion(self, emotion_probs):
        """
        Get the dominant emotion from probability distribution
        
        Args:
            emotion_probs (dict): Emotion probability distribution
            
        Returns:
            tuple: (dominant_emotion, confidence_score)
        """
        if not emotion_probs:
            return "neutral", 0.0
            
        dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])
        return dominant_emotion[0], dominant_emotion[1]
    
    def get_emotion_intensity(self, emotion_probs, emotion):
        """
        Get intensity of a specific emotion
        
        Args:
            emotion_probs (dict): Emotion probability distribution
            emotion (str): Emotion to check
            
        Returns:
            float: Intensity score (0.0 to 1.0)
        """
        return emotion_probs.get(emotion, 0.0)