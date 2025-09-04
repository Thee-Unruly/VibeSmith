# Contextual filters for emotion processing
"""
Applies contextual filters to emotions based on time, location, user history
"""
import datetime
import json
import os

class ContextFilters:
    def __init__(self, user_data_path=None):
        self.user_data_path = user_data_path
        self.user_history = self._load_user_history()
    
    def _load_user_history(self):
        """Load user emotion history if available"""
        if self.user_data_path and os.path.exists(self.user_data_path):
            try:
                with open(self.user_data_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def apply_time_context(self, emotion_probs):
        """
        Adjust emotions based on time of day
        
        Args:
            emotion_probs (dict): Emotion probabilities
            
        Returns:
            dict: Time-adjusted emotion probabilities
        """
        adjusted = emotion_probs.copy()
        hour = datetime.datetime.now().hour
        
        # Morning (6am-12pm): boost positive emotions
        if 6 <= hour < 12:
            for emotion in ['joy', 'anticipation', 'surprise']:
                if emotion in adjusted:
                    adjusted[emotion] *= 1.2
        
        # Evening (6pm-12am): boost calm/neutral emotions
        elif 18 <= hour < 24:
            for emotion in ['neutral', 'trust', 'calm']:
                if emotion in adjusted:
                    adjusted[emotion] *= 1.2
        
        # Night (12am-6am): reduce intense emotions
        elif 0 <= hour < 6:
            for emotion in ['anger', 'fear', 'surprise']:
                if emotion in adjusted:
                    adjusted[emotion] *= 0.7
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {e: p/total for e, p in adjusted.items()}
        
        return adjusted
    
    def apply_historical_context(self, emotion_probs, user_id=None):
        """
        Adjust emotions based on user's historical patterns
        
        Args:
            emotion_probs (dict): Emotion probabilities
            user_id (str): Optional user identifier
            
        Returns:
            dict: History-adjusted emotion probabilities
        """
        if not user_id or not self.user_history.get(user_id):
            return emotion_probs
        
        user_data = self.user_history[user_id]
        adjusted = emotion_probs.copy()
        
        # Get user's typical emotion pattern
        typical_emotions = user_data.get('typical_emotions', {})
        
        # If current emotion deviates significantly from typical pattern,
        # it might be more noteworthy
        for emotion, current_prob in emotion_probs.items():
            typical_prob = typical_emotions.get(emotion, 0.1)
            deviation = abs(current_prob - typical_prob)
            
            # Amplify emotions that deviate from norm
            if deviation > 0.3:
                adjusted[emotion] *= 1.3
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {e: p/total for e, p in adjusted.items()}
        
        return adjusted
    
    def save_emotion_record(self, emotion_probs, user_id=None):
        """
        Save current emotion to user history
        
        Args:
            emotion_probs (dict): Emotion probabilities to save
            user_id (str): Optional user identifier
        """
        if not user_id:
            return
        
        if user_id not in self.user_history:
            self.user_history[user_id] = {
                'emotion_history': [],
                'typical_emotions': {}
            }
        
        # Add timestamped record
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'emotions': emotion_probs
        }
        self.user_history[user_id]['emotion_history'].append(record)
        
        # Update typical emotions (simple average)
        history = self.user_history[user_id]['emotion_history']
        if len(history) > 0:
            typical = {}
            for record in history:
                for emotion, prob in record['emotions'].items():
                    typical[emotion] = typical.get(emotion, 0) + prob
            
            typical = {e: p/len(history) for e, p in typical.items()}
            self.user_history[user_id]['typical_emotions'] = typical
        
        # Save to file
        if self.user_data_path:
            with open(self.user_data_path, 'w') as f:
                json.dump(self.user_history, f, indent=2)