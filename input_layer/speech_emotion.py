# Speech emotion recognition
"""
Advanced Speech Emotion Recognition using audio processing and ML
"""
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
from scipy import signal

class SpeechEmotionRecognizer:
    # Standard emotion categories for speech
    SPEECH_EMOTIONS = [
        'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    def __init__(self, sample_rate=22050, duration=3, hop_length=512, n_mfcc=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.model = None
    
    def load_model(self, model_path):
        """
        Load a pre-trained speech emotion recognition model
        
        Args:
            model_path (str): Path to model file
        """
        # Implementation would load a trained model
        # Example: self.model = tf.keras.models.load_model(model_path)
        print(f"Would load model from {model_path}")
    
    def load_audio(self, audio_path, resample=True):
        """
        Load and preprocess audio file
        
        Args:
            audio_path (str): Path to audio file
            resample (bool): Whether to resample to target sample rate
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate if resample else None)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")
    
    def extract_features(self, audio, sr):
        """
        Extract comprehensive audio features for emotion recognition
        
        Args:
            audio (numpy array): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, 
                                   hop_length=self.hop_length)
        features['mfcc_mean'] = np.mean(mfccs.T, axis=0)
        features['mfcc_std'] = np.std(mfccs.T, axis=0)
        
        # Chroma feature
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.hop_length)
        features['chroma_mean'] = np.mean(chroma.T, axis=0)
        
        # Mel-scaled spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=self.hop_length)
        features['mel_mean'] = np.mean(mel.T, axis=0)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=self.hop_length)
        features['contrast_mean'] = np.mean(contrast.T, axis=0)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz.T, axis=0)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        features['zcr_mean'] = np.mean(zcr)
        
        # Root mean square energy
        rmse = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        features['rmse_mean'] = np.mean(rmse)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)
        features['centroid_mean'] = np.mean(centroid)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)
        features['bandwidth_mean'] = np.mean(bandwidth)
        
        # Pitch and fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'))
        features['pitch_mean'] = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
        
        return features
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file for emotion recognition
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            numpy array: Processed features ready for prediction
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        # Ensure consistent length
        if len(audio_trimmed) > self.sample_rate * self.duration:
            audio_trimmed = audio_trimmed[:self.sample_rate * self.duration]
        else:
            padding = self.sample_rate * self.duration - len(audio_trimmed)
            audio_trimmed = np.pad(audio_trimmed, (0, padding), mode='constant')
        
        # Extract features
        features = self.extract_features(audio_trimmed, sr)
        
        # Combine all features into a single array
        feature_vector = np.concatenate([
            features['mfcc_mean'],
            features['mfcc_std'],
            features['chroma_mean'],
            features['mel_mean'],
            features['contrast_mean'],
            features['tonnetz_mean'],
            [features['zcr_mean']],
            [features['rmse_mean']],
            [features['centroid_mean']],
            [features['bandwidth_mean']],
            [features['pitch_mean']]
        ])
        
        return feature_vector.reshape(1, -1)  # Add batch dimension
    
    def recognize_emotion(self, audio_path):
        """
        Recognize emotion from speech audio
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Emotion probabilities
        """
        try:
            # Preprocess audio
            features = self.preprocess_audio(audio_path)
            
            if self.model:
                # Use model for prediction
                # predictions = self.model.predict(features)[0]
                # emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.SPEECH_EMOTIONS, predictions)}
                pass
            else:
                # Simulate prediction based on audio features
                emotion_probs = self._simulate_emotion_prediction(features)
            
            return emotion_probs
            
        except Exception as e:
            print(f"Error recognizing emotion: {e}")
            return self._get_neutral_emotion()
    
    def _simulate_emotion_prediction(self, features):
        """
        Simulate emotion prediction based on audio features
        
        Args:
            features (numpy array): Audio features
            
        Returns:
            dict: Simulated emotion probabilities
        """
        # This is a simplified simulation - real implementation would use a trained model
        
        # Extract some feature characteristics that might correlate with emotions
        mfcc_std = np.std(features[0, :self.n_mfcc])  # MFCC standard deviation
        pitch = features[0, -1]  # Pitch mean
        energy = features[0, -4]  # RMS energy
        
        # Base probabilities influenced by features
        base_probs = np.ones(len(self.SPEECH_EMOTIONS))
        
        # Adjust based on feature characteristics
        if pitch > 200:  # High pitch
            base_probs[self.SPEECH_EMOTIONS.index('happy')] += 2
            base_probs[self.SPEECH_EMOTIONS.index('surprised')] += 1
        elif pitch < 100:  # Low pitch
            base_probs[self.SPEECH_EMOTIONS.index('sad')] += 2
            base_probs[self.SPEECH_EMOTIONS.index('calm')] += 1
        
        if energy > 0.1:  # High energy
            base_probs[self.SPEECH_EMOTIONS.index('angry')] += 2
            base_probs[self.SPEECH_EMOTIONS.index('happy')] += 1
        elif energy < 0.01:  # Low energy
            base_probs[self.SPEECH_EMOTIONS.index('sad')] += 2
            base_probs[self.SPEECH_EMOTIONS.index('calm')] += 1
        
        if mfcc_std > 5:  # Variable MFCCs (emotional speech)
            base_probs[self.SPEECH_EMOTIONS.index('angry')] += 1
            base_probs[self.SPEECH_EMOTIONS.index('fearful')] += 1
        else:  # Stable MFCCs (neutral/calm)
            base_probs[self.SPEECH_EMOTIONS.index('neutral')] += 2
            base_probs[self.SPEECH_EMOTIONS.index('calm')] += 1
        
        # Normalize to probabilities
        probs = base_probs / np.sum(base_probs)
        
        return {emotion: float(prob) for emotion, prob in zip(self.SPEECH_EMOTIONS, probs)}
    
    def analyze_real_time(self, audio_chunk, sr):
        """
        Analyze emotion from real-time audio chunk
        
        Args:
            audio_chunk (numpy array): Audio data
            sr (int): Sample rate
            
        Returns:
            dict: Emotion probabilities
        """
        # Save temporary audio file and analyze
        temp_path = "temp_audio.wav"
        sf.write(temp_path, audio_chunk, sr)
        result = self.recognize_emotion(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result
    
    def _get_neutral_emotion(self):
        """Return neutral emotion distribution"""
        return {emotion: 1.0/len(self.SPEECH_EMOTIONS) for emotion in self.SPEECH_EMOTIONS}
    
    def get_emotion_intensity(self, emotion_probs):
        """
        Calculate overall emotional intensity
        
        Args:
            emotion_probs (dict): Emotion probabilities
            
        Returns:
            float: Emotional intensity (0.0 to 1.0)
        """
        # Intensity is highest when one emotion dominates
        max_prob = max(emotion_probs.values())
        neutral_prob = emotion_probs.get('neutral', 0)
        
        # Intensity increases with dominance and decreases with neutrality
        intensity = max_prob * (1 - neutral_prob)
        return intensity