"""
Real-time audio recorder for speech emotion recognition
"""
import pyaudio
import wave
import threading
import time
import numpy as np
import os

class AudioRecorder:
    def __init__(self, sample_rate=22050, channels=1, chunk_size=1024, format=pyaudio.paInt16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None
    
    def start_recording(self, duration=5, output_path="recording.wav"):
        """
        Start recording audio for a specified duration
        
        Args:
            duration (int): Recording duration in seconds
            output_path (str): Path to save the recording
        """
        self.frames = []
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print(f"Recording for {duration} seconds...")
        
        # Record for the specified duration
        start_time = time.time()
        while self.is_recording and (time.time() - start_time) < duration:
            data = self.stream.read(self.chunk_size)
            self.frames.append(data)
        
        self.stop_recording(output_path)
    
    def start_realtime_recording(self, callback=None, chunk_duration=1.0):
        """
        Start real-time recording with callback for processing
        
        Args:
            callback (function): Function to call with audio data
            chunk_duration (float): Duration of each chunk in seconds
        """
        self.frames = []
        self.is_recording = True
        
        # Calculate chunks per second
        chunks_per_second = int(self.sample_rate / self.chunk_size)
        chunks_to_collect = int(chunks_per_second * chunk_duration)
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("Real-time recording started. Press Ctrl+C to stop.")
        
        chunk_count = 0
        current_chunk = []
        
        try:
            while self.is_recording:
                data = self.stream.read(self.chunk_size)
                current_chunk.append(data)
                chunk_count += 1
                
                # Process chunk when we have enough data
                if chunk_count >= chunks_to_collect:
                    if callback:
                        # Convert to numpy array
                        audio_data = b''.join(current_chunk)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        callback(audio_array, self.sample_rate)
                    
                    # Reset for next chunk
                    current_chunk = []
                    chunk_count = 0
                    
        except KeyboardInterrupt:
            print("\nStopping recording...")
        finally:
            self.stop_recording()
    
    def stop_recording(self, output_path=None):
        """Stop recording and optionally save to file"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Save to file if requested
        if output_path and self.frames:
            wf = wave.open(output_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            print(f"Recording saved to {output_path}")
    
    def record_and_analyze(self, emotion_recognizer, duration=5):
        """
        Record and analyze emotion in one call
        
        Args:
            emotion_recognizer: SpeechEmotionRecognizer instance
            duration (int): Recording duration in seconds
            
        Returns:
            dict: Emotion probabilities
        """
        temp_path = "temp_recording.wav"
        self.start_recording(duration, temp_path)
        
        # Analyze the recording
        emotion_probs = emotion_recognizer.recognize_emotion(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return emotion_probs
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'audio'):
            self.audio.terminate()