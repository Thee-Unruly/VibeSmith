"""
Blends emotions from multiple modalities with weighted averaging
"""
import numpy as np

class EmotionBlender:
    def __init__(self, modality_weights=None):
        # Default weights for each modality
        self.modality_weights = modality_weights or {
            'text': 0.4,    # Highest weight - most explicit
            'speech': 0.35, # Medium weight - vocal cues
            'face': 0.25    # Lower weight - can be misleading
        }
    
    def blend_emotions(self, text_emotions=None, speech_emotions=None, face_emotions=None):
        """
        Blend emotions from multiple modalities using weighted averaging
        
        Args:
            text_emotions (dict): Standardized text emotions
            speech_emotions (dict): Standardized speech emotions  
            face_emotions (dict): Standardized facial emotions
            
        Returns:
            dict: Blended emotion probabilities
        """
        # Initialize with neutral baseline
        blended = {e: 0.0 for e in [
            'joy', 'trust', 'fear', 'surprise', 
            'sadness', 'disgust', 'anger', 'anticipation', 'neutral'
        ]}
        total_weight = 0.0
        
        # Blend each available modality
        modalities = [
            ('text', text_emotions, self.modality_weights['text']),
            ('speech', speech_emotions, self.modality_weights['speech']),
            ('face', face_emotions, self.modality_weights['face'])
        ]
        
        for modality_name, emotions, weight in modalities:
            if emotions:
                for emotion, prob in emotions.items():
                    if emotion in blended:
                        blended[emotion] += prob * weight
                total_weight += weight
        
        # Normalize if we have any emotions
        if total_weight > 0:
            blended = {e: p/total_weight for e, p in blended.items()}
        
        return blended
    
    def detect_emotional_congruence(self, text_emotions, speech_emotions, face_emotions, threshold=0.7):
        """
        Check if emotions from different modalities are congruent
        
        Args:
            text_emotions (dict): Standardized text emotions
            speech_emotions (dict): Standardized speech emotions
            face_emotions (dict): Standardized facial emotions
            threshold (float): Confidence threshold for congruence
            
        Returns:
            bool: True if emotions are congruent, False otherwise
        """
        modalities = [
            ('Text', text_emotions),
            ('Speech', speech_emotions), 
            ('Face', face_emotions)
        ]
        
        # Get dominant emotions from each available modality
        dominant_emotions = []
        for modality_name, emotions in modalities:
            if emotions:
                dom_emotion = max(emotions.items(), key=lambda x: x[1])
                if dom_emotion[1] >= threshold:  # Only consider confident detections
                    dominant_emotions.append(dom_emotion[0])
        
        # Check congruence
        if len(dominant_emotions) < 2:
            return True  # Not enough data to determine incongruence
        
        # If all dominant emotions are the same, they're congruent
        if len(set(dominant_emotions)) == 1:
            return True
        
        # Special case: if text says "neutral" but other modalities detect emotion
        if (text_emotions and max(text_emotions.items(), key=lambda x: x[1])[0] == 'neutral' and
            len([e for e in dominant_emotions if e != 'neutral']) > 0):
            return False
        
        return False