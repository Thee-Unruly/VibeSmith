# Standardize emotions, probability blending
"""
Standardizes emotions from different modalities to a common framework
"""
class EmotionStandardizer:
    # Standard emotion categories (Plutchik's wheel + neutral)
    STANDARD_EMOTIONS = [
        'joy', 'trust', 'fear', 'surprise', 
        'sadness', 'disgust', 'anger', 'anticipation', 'neutral'
    ]
    
    # Mapping from different models to standard emotions
    MODEL_MAPPINGS = {
        # Text model (j-hartmann) mapping
        'text': {
            'anger': 'anger',
            'disgust': 'disgust', 
            'fear': 'fear',
            'joy': 'joy',
            'neutral': 'neutral',
            'sadness': 'sadness',
            'surprise': 'surprise'
        },
        # Speech model (RAVDESS) mapping  
        'speech': {
            'neutral': 'neutral',
            'calm': 'calm',
            'happy': 'joy',
            'sad': 'sadness',
            'angry': 'anger',
            'fearful': 'fear',
            'disgust': 'disgust',
            'surprised': 'surprise'
        },
        # Face model (FER2013) mapping
        'face': {
            'angry': 'anger',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'joy',
            'sad': 'sadness',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
    }
    
    def __init__(self):
        self.standard_emotions = {e: 0.0 for e in self.STANDARD_EMOTIONS}
    
    def standardize_emotion(self, emotion_probs, modality):
        """
        Convert modality-specific emotions to standard framework
        
        Args:
            emotion_probs (dict): Raw emotion probabilities from a modality
            modality (str): 'text', 'speech', or 'face'
            
        Returns:
            dict: Standardized emotion probabilities
        """
        if modality not in self.MODEL_MAPPINGS:
            raise ValueError(f"Unknown modality: {modality}")
        
        standardized = {e: 0.0 for e in self.STANDARD_EMOTIONS}
        
        # Map emotions according to modality
        for source_emotion, prob in emotion_probs.items():
            if source_emotion in self.MODEL_MAPPINGS[modality]:
                target_emotion = self.MODEL_MAPPINGS[modality][source_emotion]
                standardized[target_emotion] += prob
        
        # Normalize to ensure sum = 1.0
        total = sum(standardized.values())
        if total > 0:
            standardized = {e: p/total for e, p in standardized.items()}
        
        return standardized
    
    def get_dominant_emotion(self, emotion_probs):
        """
        Extract the dominant emotion from probabilities
        
        Args:
            emotion_probs (dict): Emotion probability distribution
            
        Returns:
            tuple: (dominant_emotion, confidence)
        """
        if not emotion_probs:
            return "neutral", 0.0
        
        dominant = max(emotion_probs.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]