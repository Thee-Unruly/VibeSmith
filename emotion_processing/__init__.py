"""
Main emotion processing module that coordinates standardization, blending, and context filtering
"""
from .standardizer import EmotionStandardizer
from .emotion_blender import EmotionBlender
from .context_filters import ContextFilters

class EmotionProcessor:
    def __init__(self, user_data_path=None):
        self.standardizer = EmotionStandardizer()
        self.blender = EmotionBlender()
        self.context_filters = ContextFilters(user_data_path)
    
    def process_emotions(self, text_emotions=None, speech_emotions=None, 
                        face_emotions=None, user_id=None):
        """
        Main method to process emotions from multiple modalities
        
        Args:
            text_emotions (dict): Raw text emotions
            speech_emotions (dict): Raw speech emotions
            face_emotions (dict): Raw facial emotions
            user_id (str): Optional user identifier for personalization
            
        Returns:
            dict: Processed and contextualized emotion probabilities
        """
        # Standardize emotions from each modality
        std_text = (self.standardizer.standardize_emotion(text_emotions, 'text') 
                   if text_emotions else None)
        std_speech = (self.standardizer.standardize_emotion(speech_emotions, 'speech') 
                     if speech_emotions else None)
        std_face = (self.standardizer.standardize_emotion(face_emotions, 'face') 
                   if face_emotions else None)
        
        # Blend emotions from available modalities
        blended = self.blender.blend_emotions(std_text, std_speech, std_face)
        
        # Apply context filters
        time_adjusted = self.context_filters.apply_time_context(blended)
        history_adjusted = self.context_filters.apply_historical_context(time_adjusted, user_id)
        
        # Save to history
        self.context_filters.save_emotion_record(history_adjusted, user_id)
        
        # Check for emotional congruence
        congruent = self.blender.detect_emotional_congruence(std_text, std_speech, std_face)
        
        return {
            'emotions': history_adjusted,
            'congruent': congruent,
            'dominant_emotion': self.standardizer.get_dominant_emotion(history_adjusted)
        }