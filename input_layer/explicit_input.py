# User mood selection/input
"""
Module for handling explicit user emotion input
"""
class ExplicitInput:
    # Standard emotion categories based on Plutchik's wheel of emotions
    EMOTION_CATEGORIES = [
        'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation'
    ]
    
    EMOTION_INTENSITY = {
        'low': 0.3,
        'medium': 0.6, 
        'high': 0.9
    }
    
    def __init__(self):
        self.user_emotion = None
        self.intensity = None
    
    def get_emotion_from_selection(self, emotion, intensity='medium'):
        """
        Get emotion from user's explicit selection
        
        Args:
            emotion (str): One of the EMOTION_CATEGORIES
            intensity (str): 'low', 'medium', or 'high'
            
        Returns:
            dict: Emotion with probability score
        """
        if emotion.lower() not in self.EMOTION_CATEGORIES:
            raise ValueError(f"Emotion must be one of: {self.EMOTION_CATEGORIES}")
            
        if intensity.lower() not in self.EMOTION_INTENSITY:
            raise ValueError("Intensity must be 'low', 'medium', or 'high'")
            
        self.user_emotion = emotion.lower()
        self.intensity = self.EMOTION_INTENSITY[intensity.lower()]
        
        # Return as probability distribution (focused on the selected emotion)
        emotion_dict = {e: 0.01 for e in self.EMOTION_CATEGORIES}  # Small baseline
        emotion_dict[self.user_emotion] = self.intensity
        
        return emotion_dict
    
    def get_emotion_from_text(self, text_input):
        """
        Parse emotion from text input like "I'm feeling happy today"
        
        Args:
            text_input (str): User's text description of their emotion
            
        Returns:
            dict: Emotion probabilities
        """
        emotion_dict = {e: 0.01 for e in self.EMOTION_CATEGORIES}
        
        text_lower = text_input.lower()
        
        # Simple keyword matching - in production, use a more sophisticated approach
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'awesome'],
            'sadness': ['sad', 'unhappy', 'depressed', 'down', 'miserable'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'fear', 'nervous', 'anxious'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'trust': ['trust', 'confident', 'secure', 'safe'],
            'disgust': ['disgust', 'disgusted', 'revolted', 'sickened'],
            'anticipation': ['anticipate', 'expect', 'looking forward', 'eager']
        }
        
        # Count keyword matches
        matches = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                matches[emotion] = count
        
        # If we found matches, distribute probabilities
        if matches:
            total = sum(matches.values())
            for emotion, count in matches.items():
                emotion_dict[emotion] = count / total * 0.9  # Scale to 0.9 max
        else:
            # Default to neutral if no emotion detected
            emotion_dict = {e: 0.125 for e in self.EMOTION_CATEGORIES}  # Even distribution
        
        return emotion_dict