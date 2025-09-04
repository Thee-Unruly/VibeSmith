# Facial emotion recognition
"""
Advanced Facial Emotion Detection using Deep Learning
"""
import numpy as np
import cv2
from PIL import Image
import os

class FacialEmotionDetector:
    # Emotion mapping for FER2013 dataset
    EMOTION_MAP = {
        0: 'anger',
        1: 'disgust', 
        2: 'fear',
        3: 'joy',
        4: 'sadness',
        5: 'surprise',
        6: 'neutral'
    }
    
    def __init__(self, use_pretrained=True):
        self.face_cascade = None
        self.emotion_model = None
        self.input_size = (48, 48)  # Standard size for emotion models
        
        if use_pretrained:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize face detection and emotion classification models"""
        try:
            # Initialize Haar Cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # In a real implementation, we would load a pre-trained emotion model
            # For example: self.emotion_model = load_keras_model('fer2013_model.h5')
            print("Face detection model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.face_cascade = None
    
    def load_custom_model(self, model_path):
        """
        Load a custom emotion classification model
        
        Args:
            model_path (str): Path to model file
        """
        # Implementation for loading custom models
        # self.emotion_model = tf.keras.models.load_model(model_path)
        print(f"Would load custom model from {model_path}")
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image using OpenCV
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            list: Detected faces with coordinates and confidence
        """
        if not self.face_cascade:
            raise ValueError("Face detection model not initialized")
        
        try:
            # Read and convert image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of dictionaries
            detected_faces = []
            for (x, y, w, h) in faces:
                detected_faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,  # Haar cascade doesn't provide confidence
                    'region': gray[y:y+h, x:x+w]  # Face region
                })
            
            return detected_faces
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def preprocess_face(self, face_region):
        """
        Preprocess face region for emotion classification
        
        Args:
            face_region (numpy array): Grayscale face image
            
        Returns:
            numpy array: Preprocessed face ready for prediction
        """
        # Resize to model input size
        resized = cv2.resize(face_region, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        processed = np.expand_dims(normalized, axis=0)  # Add batch dimension
        processed = np.expand_dims(processed, axis=-1)  # Add channel dimension
        
        return processed
    
    def predict_emotion(self, processed_face):
        """
        Predict emotion from preprocessed face
        
        Args:
            processed_face (numpy array): Preprocessed face image
            
        Returns:
            dict: Emotion probabilities
        """
        # In a real implementation:
        # predictions = self.emotion_model.predict(processed_face)[0]
        # emotion_probs = {self.EMOTION_MAP[i]: float(pred) for i, pred in enumerate(predictions)}
        
        # Simulated prediction for demonstration
        emotions = list(self.EMOTION_MAP.values())
        simulated_probs = np.random.dirichlet(np.ones(len(emotions)), size=1)[0]
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(emotions, simulated_probs)}
        
        return emotion_probs
    
    def analyze_emotion(self, image_path, use_dominant_face=True):
        """
        Analyze emotion from facial expression in image
        
        Args:
            image_path (str): Path to image file
            use_dominant_face (bool): If True, analyze only the largest face
            
        Returns:
            dict: Emotion probabilities or list of emotions for multiple faces
        """
        # Detect faces
        faces = self.detect_faces(image_path)
        
        if not faces:
            print("No faces detected in the image")
            return self._get_neutral_emotion()
        
        # Analyze each face
        face_emotions = []
        for i, face in enumerate(faces):
            try:
                # Preprocess face
                processed_face = self.preprocess_face(face['region'])
                
                # Predict emotion
                emotion_probs = self.predict_emotion(processed_face)
                face_emotions.append({
                    'face_id': i,
                    'bbox': face['bbox'],
                    'emotions': emotion_probs,
                    'dominant_emotion': max(emotion_probs.items(), key=lambda x: x[1])[0]
                })
                
            except Exception as e:
                print(f"Error analyzing face {i}: {e}")
                continue
        
        if not face_emotions:
            return self._get_neutral_emotion()
        
        # Return based on preference
        if use_dominant_face:
            # Find the largest face (assuming size correlates with importance)
            largest_face = max(face_emotions, key=lambda x: x['bbox'][2] * x['bbox'][3])
            return largest_face['emotions']
        else:
            # Return average of all faces
            return self._average_emotions([fe['emotions'] for fe in face_emotions])
    
    def analyze_emotion_from_array(self, image_array):
        """
        Analyze emotion from image array (for real-time applications)
        
        Args:
            image_array (numpy array): Image data
            
        Returns:
            dict: Emotion probabilities
        """
        # Save temporary image and analyze
        temp_path = "temp_face_image.jpg"
        cv2.imwrite(temp_path, image_array)
        result = self.analyze_emotion(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result
    
    def _average_emotions(self, emotion_list):
        """
        Average multiple emotion probability distributions
        
        Args:
            emotion_list (list): List of emotion probability dictionaries
            
        Returns:
            dict: Averaged emotion probabilities
        """
        if not emotion_list:
            return self._get_neutral_emotion()
        
        # Initialize average
        avg_emotions = {emotion: 0.0 for emotion in self.EMOTION_MAP.values()}
        
        # Sum probabilities
        for emotions in emotion_list:
            for emotion, prob in emotions.items():
                avg_emotions[emotion] += prob
        
        # Normalize
        total = sum(avg_emotions.values())
        if total > 0:
            avg_emotions = {emotion: prob/total for emotion, prob in avg_emotions.items()}
        
        return avg_emotions
    
    def _get_neutral_emotion(self):
        """Return neutral emotion distribution"""
        emotions = list(self.EMOTION_MAP.values())
        return {emotion: 1.0/len(emotions) for emotion in emotions}
    
    def visualize_detection(self, image_path, output_path=None):
        """
        Visualize face detection and emotion results
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image (optional)
            
        Returns:
            numpy array: Image with detection visualization
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Detect and analyze faces
        faces = self.detect_faces(image_path)
        face_emotions = []
        
        for face in faces:
            processed_face = self.preprocess_face(face['region'])
            emotion_probs = self.predict_emotion(processed_face)
            dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
            face_emotions.append((face['bbox'], dominant_emotion))
        
        # Draw bounding boxes and labels
        for (x, y, w, h), emotion in face_emotions:
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label background
            cv2.rectangle(image, (x, y-25), (x+w, y), (0, 255, 0), -1)
            
            # Draw emotion label
            cv2.putText(image, emotion, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save or return image
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image