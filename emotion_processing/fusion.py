"""
Multi-modal emotion fusion combining text, speech, and facial emotions
"""
import numpy as np

class EmotionFusion:
    def __init__(self, weights=None):
        # Default weights for each modality
        self.weights = weights or {
            'text': 0.4,
            'speech': 0.3, 
            'face': 0.3
        }
        
        # Emotion mapping between different modalities
        self.emotion_mapping = {
            'text': {
                'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
                'joy': 'happy', 'neutral': 'neutral', 'sadness': 'sad',
                'surprise': 'surprise'
            },
            'speech': {
                'neutral': 'neutral', 'calm': 'neutral', 'happy': 'happy',
                'sad': 'sad', 'angry': 'angry', 'fearful': 'fear',
                'disgust': 'disgust', 'surprised': 'surprise'
            },
            'face': {
                'angry': 'angry', 'disgust': 'disgust', 'fear': 'fear',
                'happy': 'happy', 'sad': 'sad', 'surprise': 'surprise',
                'neutral': 'neutral'
            }
        }
    
    def fuse_emotions(self, text_emotions, speech_emotions, face_emotions):
        """
        Fuse emotions from multiple modalities
        
        Args:
            text_emotions (dict): Text emotion probabilities
            speech_emotions (dict): Speech emotion probabilities  
            face_emotions (dict): Facial emotion probabilities
            
        Returns:
            dict: Fused emotion probabilities
        """
        # Map all emotions to a common set
        common_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize fused probabilities
        fused_probs = {emotion: 0.0 for emotion in common_emotions}
        
        # Process each modality
        modalities = [
            ('text', text_emotions),
            ('speech', speech_emotions), 
            ('face', face_emotions)
        ]
        
        for modality_name, emotions in modalities:
            if emotions:
                weight = self.weights[modality_name]
                mapped_emotions = self._map_emotions(modality_name, emotions, common_emotions)
                
                for emotion, prob in mapped_emotions.items():
                    fused_probs[emotion] += prob * weight
        
        # Normalize
        total = sum(fused_probs.values())
        if total > 0:
            fused_probs = {emotion: prob/total for emotion, prob in fused_probs.items()}
        
        return fused_probs
    
    def _map_emotions(self, modality, emotions, target_emotions):
        """
        Map emotions from a modality to the target emotion set
        
        Args:
            modality (str): Modality name
            emotions (dict): Original emotion probabilities
            target_emotions (list): Target emotion names
            
        Returns:
            dict: Mapped emotion probabilities
        """
        mapped = {emotion: 0.0 for emotion in target_emotions}
        
        for src_emotion, prob in emotions.items():
            # Map to target emotion
            if modality in self.emotion_mapping and src_emotion in self.emotion_mapping[modality]:
                target_emotion = self.emotion_mapping[modality][src_emotion]
                if target_emotion in mapped:
                    mapped[target_emotion] += prob
            else:
                # Default to neutral if mapping not found
                mapped['neutral'] += prob
        
        return mapped
    
    def detect_emotional_congruence(self, text_emotions, speech_emotions, face_emotions, threshold=0.7):
        """
        Detect if emotions from different modalities are congruent
        
        Args:
            text_emotions (dict): Text emotion probabilities
            speech_emotions (dict): Speech emotion probabilities
            face_emotions (dict): Facial emotion probabilities
            threshold (float): Congruence threshold
            
        Returns:
            bool: True if emotions are congruent, False otherwise
        """
        # Get dominant emotions from each modality
        modalities = [
            ('text', text_emotions),
            ('speech', speech_emotions),
            ('face', face_emotions)
        ]
        
        dominant_emotions = []
        for modality_name, emotions in modalities:
            if emotions:
                dom_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                # Map to common emotion
                if modality_name in self.emotion_mapping and dom_emotion in self.emotion_mapping[modality_name]:
                    mapped_emotion = self.emotion_mapping[modality_name][dom_emotion]
                    dominant_emotions.append(mapped_emotion)
        
        # Check if all dominant emotions are the same
        if len(set(dominant_emotions)) == 1:
            return True, dominant_emotions[0]
        
        # Check if at least two modalities agree
        emotion_counts = {}
        for emotion in dominant_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        max_count = max(emotion_counts.values())
        if max_count >= 2 and len(dominant_emotions) >= 2:
            # Get the emotion with max count
            congruent_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            return True, congruent_emotion
        
        return False, "incongruent"