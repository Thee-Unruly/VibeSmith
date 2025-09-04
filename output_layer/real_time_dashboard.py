"""
Real-time emotion visualization dashboard
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time

class RealTimeEmotionDashboard:
    def __init__(self, emotions):
        self.emotions = emotions
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bar_containers = None
        self.current_emotions = {emotion: 0.0 for emotion in emotions}
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the initial plot"""
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Real-Time Emotion Analysis')
        self.ax.set_xticks(range(len(self.emotions)))
        self.ax.set_xticklabels(self.emotions, rotation=45)
        
        # Initial bars
        bars = self.ax.bar(range(len(self.emotions)), 
                          [0] * len(self.emotions),
                          color='skyblue',
                          alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        '%.2f' % height,
                        ha='center', va='bottom')
        
        self.bar_containers = bars
        plt.tight_layout()
    
    def update_emotions(self, emotion_probs):
        """Update the current emotion probabilities"""
        for emotion, prob in emotion_probs.items():
            if emotion in self.current_emotions:
                self.current_emotions[emotion] = prob
    
    def update_plot(self, frame):
        """Update the plot with current emotions"""
        for i, emotion in enumerate(self.emotions):
            # Update bar height
            self.bar_containers[i].set_height(self.current_emotions[emotion])
            
            # Update text label
            for text in self.ax.texts:
                if text.get_position()[0] == self.bar_containers[i].get_x() + self.bar_containers[i].get_width()/2.:
                    text.set_text('%.2f' % self.current_emotions[emotion])
                    text.set_position((text.get_position()[0], self.current_emotions[emotion]))
                    break
        
        return self.bar_containers
    
    def start_dashboard(self):
        """Start the real-time dashboard"""
        ani = FuncAnimation(self.fig, self.update_plot, interval=500, blit=True)
        plt.show()

# Simple console-based real-time display
class ConsoleEmotionDisplay:
    def __init__(self, emotions):
        self.emotions = emotions
        self.current_emotions = {emotion: 0.0 for emotion in emotions}
    
    def update_display(self, emotion_probs):
        """Update and display emotions in console"""
        self.current_emotions.update(emotion_probs)
        
        # Clear console (works on most systems)
        print("\033[H\033[J")  # Clear console
        
        print("=== REAL-TIME EMOTION ANALYSIS ===")
        print("Press Ctrl+C to stop\n")
        
        # Display emotions as bars
        for emotion in self.emotions:
            prob = self.current_emotions.get(emotion, 0.0)
            bar_length = int(prob * 40)  # Scale to 40 characters
            bar = "█" * bar_length + " " * (40 - bar_length)
            print(f"{emotion:12} [{bar}] {prob:.3f}")
        
        # Display dominant emotion
        dominant_emotion = max(self.current_emotions.items(), key=lambda x: x[1])
        print(f"\nDominant emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.3f})")