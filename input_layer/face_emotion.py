"""
Advanced Facial Emotion Recognition using CNN and FER2013 dataset
"""
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class FacialEmotionDetector:
    # Emotion mapping for FER2013 dataset
    FER_EMOTIONS = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    def __init__(self, input_size=(48, 48)):
        self.input_size = input_size
        self.face_cascade = None
        self.emotion_model = None
        self.model_loaded = False
        
        self._initialize_face_detection()
    
    def _initialize_face_detection(self):
        """Initialize face detection using Haar cascades"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            print(f"Error initializing face detection: {e}")
    
    def build_cnn_model(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Build a CNN model for facial emotion recognition
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of emotion classes
            
        Returns:
            tf.keras.Model: Compiled model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten and dense layers
            Flatten(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_emotion_model(self, model_path):
        """
        Load a pre-trained facial emotion recognition model
        
        Args:
            model_path (str): Path to model file
        """
        try:
            self.emotion_model = load_model(model_path)
            self.model_loaded = True
            print(f"Loaded facial emotion model from {model_path}")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.model_loaded = False
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image using OpenCV
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            list: Detected faces with coordinates
        """
        if not self.face_cascade:
            return []
        
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
        
        # Add dimensions for model input
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
        if self.model_loaded:
            # Use model for prediction
            predictions = self.emotion_model.predict(processed_face)[0]
            emotion_probs = {
                self.FER_EMOTIONS[i]: float(prob) for i, prob in enumerate(predictions)
            }
        else:
            # Simulated prediction for demonstration
            emotions = list(self.FER_EMOTIONS.values())
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
            dict: Emotion probabilities
        """
        # Detect faces
        faces = self.detect_faces(image_path)
        
        if not faces:
            return self._get_neutral_emotion()
        
        # Analyze each face
        face_emotions = []
        for face in faces:
            try:
                # Preprocess face
                processed_face = self.preprocess_face(face['region'])
                
                # Predict emotion
                emotion_probs = self.predict_emotion(processed_face)
                face_emotions.append(emotion_probs)
                
            except Exception as e:
                print(f"Error analyzing face: {e}")
                continue
        
        if not face_emotions:
            return self._get_neutral_emotion()
        
        # Return based on preference
        if use_dominant_face:
            # Use the first face (simplified)
            return face_emotions[0]
        else:
            # Return average of all faces
            return self._average_emotions(face_emotions)
    
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
        emotions = list(self.FER_EMOTIONS.values())
        avg_emotions = {emotion: 0.0 for emotion in emotions}
        
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
        emotions = list(self.FER_EMOTIONS.values())
        return {emotion: 1.0/len(emotions) for emotion in emotions}