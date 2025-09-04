"""
Real-time Speech Emotion Recognition with live audio processing
"""
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
import tempfile
from .audio_recorder import AudioRecorder

class SpeechEmotionRecognizer:
    # Emotion mapping for RAVDESS dataset
    RAVDESS_EMOTIONS = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    def __init__(self, sample_rate=22050, chunk_duration=3.0, n_mfcc=40, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.recorder = AudioRecorder(sample_rate=sample_rate)
        self.model = None
        self.model_loaded = False
    
    def start_realtime_analysis(self, callback=None):
        """
        Start real-time emotion analysis
        
        Args:
            callback (function): Function to call with emotion results
        """
        def process_audio_chunk(audio_data, sample_rate):
            """Process audio chunk and detect emotions"""
            # Save temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                # Save audio to temporary file
                sf.write(temp_path, audio_data, sample_rate)
                
                # Extract features and predict
                features = self.extract_mfcc_features(temp_path)
                if features is not None:
                    emotion_probs = self.predict_emotion(features)
                    
                    # Call callback with results
                    if callback:
                        callback(emotion_probs)
                        
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Start real-time recording
        self.recorder.start_realtime_recording(process_audio_chunk, self.chunk_duration)
    
    def extract_mfcc_features(self, audio_path):
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            numpy array: MFCC features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Ensure we have enough audio
            if len(y_trimmed) < self.sample_rate * 0.5:  # At least 0.5 seconds
                return None
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=y_trimmed, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            
            return features.T  # Transpose to (time_steps, features)
            
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return None
    
    def predict_emotion(self, features):
        """
        Predict emotion from features
        
        Args:
            features (numpy array): Audio features
            
        Returns:
            dict: Emotion probabilities
        """
        if self.model_loaded:
            # Use model for prediction
            features_processed = self.preprocess_features(features)
            predictions = self.model.predict(features_processed)[0]
            emotion_probs = {
                emotion: float(prob) for emotion, prob in 
                zip(self.RAVDESS_EMOTIONS.values(), predictions)
            }
        else:
            # Simulate prediction based on audio features
            emotion_probs = self._simulate_emotion_prediction(features)
        
        return emotion_probs
    
    def preprocess_features(self, features):
        """
        Preprocess features for model input
        
        Args:
            features (numpy array): Raw features
            
        Returns:
            numpy array: Processed features
        """
        # Standardize features
        if len(features) > 0:
            features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Pad or truncate to fixed size
        target_length = 130  # Typical length for 3 seconds of audio
        if features.shape[0] > target_length:
            features = features[:target_length, :]
        else:
            padding = target_length - features.shape[0]
            features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        
        # Reshape for model input
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        return features
    
    def record_and_analyze(self, duration=5):
        """
        Record audio and analyze emotion
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            dict: Emotion probabilities
        """
        return self.recorder.record_and_analyze(self, duration)
    
    def _simulate_emotion_prediction(self, features):
        """
        Simulate emotion prediction based on audio features
        
        Args:
            features (numpy array): Audio features
            
        Returns:
            dict: Simulated emotion probabilities
        """
        # Extract feature characteristics for simulation
        if features is None or len(features) == 0:
            return self._get_neutral_emotion()
        
        # Calculate various audio properties for simulation
        mfcc_std = np.std(features[:, :self.n_mfcc]) if len(features) > 0 else 1.0
        energy = np.mean(np.abs(features[:, :self.n_mfcc])) if len(features) > 0 else 0.5
        
        # Base probabilities
        emotions = list(self.RAVDESS_EMOTIONS.values())
        base_probs = np.ones(len(emotions))
        
        # Adjust probabilities based on audio characteristics
        if energy > 0.8:  # High energy
            base_probs[emotions.index('angry')] += 3
            base_probs[emotions.index('happy')] += 2
            base_probs[emotions.index('surprised')] += 1
        elif energy < 0.3:  # Low energy
            base_probs[emotions.index('sad')] += 3
            base_probs[emotions.index('calm')] += 2
            base_probs[emotions.index('neutral')] += 1
        
        if mfcc_std > 2.0:  # Highly variable (emotional)
            base_probs[emotions.index('angry')] += 2
            base_probs[emotions.index('fearful')] += 1.5
            base_probs[emotions.index('surprised')] += 1
        else:  # Stable (calm/neutral)
            base_probs[emotions.index('neutral')] += 2
            base_probs[emotions.index('calm')] += 1.5
        
        # Normalize to probabilities
        probs = base_probs / np.sum(base_probs)
        
        return {emotion: float(prob) for emotion, prob in zip(emotions, probs)}
    
    def _get_neutral_emotion(self):
        """Return neutral emotion distribution"""
        emotions = list(self.RAVDESS_EMOTIONS.values())
        return {emotion: 1.0/len(emotions) for emotion in emotions}
    
    def stop_realtime_analysis(self):
        """Stop real-time analysis"""
        self.recorder.stop_recording()