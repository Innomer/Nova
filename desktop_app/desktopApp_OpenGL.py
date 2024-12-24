import sys
import threading
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QLabel
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from OpenGL.GL import *
from noise import pnoise2
import pyaudio
import speech_recognition as sr
from math import cos, sin, radians
from queue import Queue
import time
from matplotlib import pyplot as plt
import sounddevice as sd 
import soundfile as sf
from voice_auth import feature_extraction, verify_speaker
import noisereduce as nr
import os
class Particle:
    def __init__(self):
        self.angle = np.random.uniform(0, 360)
        self.radius = np.random.uniform(0.3, 0.8)
        self.speed = np.random.uniform(0.1, 1.0)
        self.history = deque(maxlen=10)  # Store positions for trails

        self.x_offset = np.random.uniform(0, 0.5)
        self.y_offset = np.random.uniform(0, 0.5)
        
        self.color=(0.0, 0.8, 1.0)
        self.trail_color=(0.5, 0.8, 1.0)

    def update(self, loudness):
        # Update radius and speed based on loudness
        self.radius = 0.1 + loudness / 2500.0
        self.speed = 0.1 + loudness / 25000.0

        # Update angle
        self.angle += self.speed
        if self.angle >= 360:
            self.angle -= 360

        # Calculate new position
        angle_rad = radians(self.angle)
        x = self.radius * cos(angle_rad) + pnoise2(self.x_offset, self.y_offset)
        y = self.radius * sin(angle_rad) + pnoise2(self.x_offset + 0.5, self.y_offset + 0.5)

        # Update Perlin noise offsets for smooth movement
        self.x_offset += 0.01
        self.y_offset += 0.01

        # Add the current position to history
        self.history.append((x, y))
    
    def update_color(self, p_color, t_color):
        self.color=p_color
        self.trail_color=t_color

    def draw(self):
        # Draw trails
        glBegin(GL_LINE_STRIP)
        for i, (x, y) in enumerate(self.history):
            alpha = (i + 1) / len(self.history)  # Gradual fade
            glColor4f(self.trail_color[0], self.trail_color[1], self.trail_color[2], alpha) # Trail color
            glVertex2f(x, y)
        glEnd()

        # Draw the particle
        glColor3f(self.color[0], self.color[1], self.color[2])
        glBegin(GL_POINTS)
        x, y = self.history[-1]
        glVertex2f(x, y)
        glEnd()

class AudioThread(QThread):
    loudness_signal = pyqtSignal(float)  # Signal to pass loudness to the main thread
    audio_signal = pyqtSignal(np.ndarray)  # Signal to send audio data to the VoiceRecognitionThread

    def __init__(self):
        super().__init__()

    def run(self):
        """Capture audio in real-time and calculate loudness."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)

        print("Real-time audio stream is active.")

        while True:
            try:
                # Read audio data from the stream
                data = stream.read(1024, exception_on_overflow=False)

                # Convert audio data to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Calculate RMS loudness
                rms = np.abs(audio_data).mean()
                # print(rms)
                self.loudness_signal.emit(rms)
                self.audio_signal.emit(audio_data)

            except Exception as e:
                print(f"Audio streaming error: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

class VoiceRecognitionThread(QThread):
    speech_text_signal = pyqtSignal(str)  # Signal to pass text to the main thread
    pass_audio_signal = pyqtSignal(np.ndarray)  # Signal to pass audio data to GeneralFunctionalityThread

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.queue = Queue()  # Queue to receive audio data from AudioThread
        self.loudness = 0.0
        self.activation_threshold = 1000
        self.audio_buffer = deque(maxlen=44100 * 10)  # Circular buffer for 10 seconds of audio (44100 samples/sec)
        self.processing = False  # Whether we are currently processing a loudness-triggered segment
        self.loudness_start_time = None
        self.margin_seconds = 2  # Margin of audio to include before activation (2 seconds)
        self.sample_counter = 0  
        self.start_sample_index = None
        self.end_sample_index = None
        
        self.intent = None
        self.general_functionality_thread = None
        self.registered_once = False
        self.logged_in=False

        # Set up a live plot for dequeued audio data
        # self.fig, self.ax = plt.subplots()
        # self.plot_line, = self.ax.plot([], [], lw=1)
        # plt.ion()
        # plt.show()
        
    def update_loudness(self, loudness):
        self.loudness = loudness

    def run(self):
        """Process incoming audio data for speech recognition."""
        while True:
            try:
                # Wait for audio data from the AudioThread
                audio_data = self.queue.get()

                # Continuously add audio data to the circular buffer
                self.audio_buffer.extend(audio_data)
                self.sample_counter += len(audio_data)

                # Handle loudness-triggered audio processing
                if self.loudness >= self.activation_threshold:
                    if not self.processing:
                        self.processing = True
                        self.loudness_start_time = time.time()
                        self.start_sample_index = self.sample_counter
                        print("Loudness above threshold, starting to track segment...")

                elif self.processing:
                    # Stop processing if loudness is low for 5 seconds
                    if time.time() - self.loudness_start_time > 5:
                        self.processing = False
                        self.end_sample_index = self.sample_counter
                        print("Loudness below threshold, extracting audio segment...")

                        # Extract relevant portion of the audio buffer
                        margin_samples = int(self.margin_seconds * 44100)
                        segment = self.get_relevant_audio(margin_samples)
                        
                        # Reset
                        self.start_sample_index = None
                        self.end_sample_index = None
                        self.audio_buffer.clear()

                        # Plot, play, and recognize speech from the segment
                        # self.plot_audio_buffer(segment)
                        # self.play_audio(segment)
                        self.recognize_speech(segment)

            except Exception as e:
                print(f"Error in VoiceRecognitionThread: {e}")

    def get_relevant_audio(self, margin_samples):
        """Extract the relevant portion of the buffer."""
        # Ensure start and end indices are valid
        if self.start_sample_index is None or self.end_sample_index is None:
            return np.array([])
        
        buffer_list = list(self.audio_buffer)
        
        # Calculate start and end sample indices with margin
        start_index = max(0, self.start_sample_index - margin_samples - self.sample_counter + len(self.audio_buffer))
        end_index = min(len(self.audio_buffer), self.end_sample_index - self.sample_counter + len(self.audio_buffer) + margin_samples)

        print(f"Start index: {start_index}, End index: {end_index}")
        
        return np.array(buffer_list[start_index:end_index])

    def plot_audio_buffer(self, audio_array):
        """Plot the audio data in the buffer."""
        self.ax.clear()  # Clear previous plot
        self.ax.plot(audio_array, lw=1, color='blue')
        self.ax.set_title("Audio Segment Data")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def play_audio(self, audio_array):
        """Play the extracted audio segment."""
        print("Playing extracted audio segment...")
        # Normalize audio array to fit within the range of int16
        audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        sd.play(audio_array, samplerate=44100)
        sd.wait()  # Wait until playback is complete

    def set_general_functionality_thread(self, thread):
        """Set reference to GeneralFunctionalityThread to call methods from here."""
        self.general_functionality_thread = thread
        
    def recognize_speech(self, audio_array):
        """Recognize speech from the extracted audio."""
        try:
            print("Processing speech recognition...")
            audio_bytes = audio_array.tobytes()
            audio = sr.AudioData(audio_bytes, 44100, 2)
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized Speech: {text}")

            self.speech_text_signal.emit(text)
            
            if "register" in text.lower() and not self.registered_once:
                self.intent = "register"
                print("Intent: Register")
                self.pass_audio_signal.emit(audio_array)
                self.registered_once = True
            elif "register" in text.lower() and self.registered_once:
                if self.logged_in:
                    self.intent = "register"
                    print("Intent: Register")
                    self.pass_audio_signal.emit(audio_array)
                else:
                    print("Restricted Access") # Remind user to login
            elif "login" in text.lower():
                self.intent = "login"
                print("Intent: Login")
                self.pass_audio_signal.emit(audio_array)
            elif "logout" in text.lower():
                self.intent = "logout"
                print("Intent: Logout")
                self.logged_in=False
                self.pass_audio_signal.emit(audio_array)
            else:
                if self.logged_in:
                    print("Intent: Other")
                    pass # Allow access to other stuff
                else:
                    print("Restricted Access")
                    pass # Remind user to login

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

class GeneralFunctionalityThread(QThread):
    verification_result_signal = pyqtSignal(int)  # Signal to pass verification result to the main thread
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.voice_thread=None
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("public", exist_ok=True) # Add the model in the public folder

    def run(self):
        """Thread entry point, runs continuously."""
        pass  # Can be expanded to include other functionalities as needed

    def process_audio_clip(self, audio_data):
        if self.voice_thread:
            if self.voice_thread.intent == "register":
                self.save_audio_clip(audio_data)
            elif self.voice_thread.intent == "login":
                self.compare_audio_clip(audio_data)
            elif self.voice_thread.intent == "logout":
                self.voice_thread.logged_in=False
                self.verification_result_signal.emit(1)
                print("Logged out successfully!")
    
    def save_audio_clip(self,audio_data):
        print("Saving received voice clip...")
        audio_array = np.array(audio_data)

        # Normalize and save audio as WAV file
        audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        file_name = f"recordings/voice_clip_{int(time.time())}.wav"       
        # Create a thread for feature extraction
        def extract_features():
            print("Reducing noise and saving audio clip...")
            audio_array_reduced = nr.reduce_noise(y=audio_array, sr=44100)
            sf.write(file_name, audio_array_reduced, samplerate=44100)        
            print(f"Voice clip saved as: {file_name}")
            print("Extracting features...")
            voiced_features = feature_extraction(file_name, fs=44100)
            with open("recordings/user_voice_features.npy", "wb") as f:
                np.save(f, voiced_features)
            os.remove(file_name)
            print("Voice features saved as: user_voice_feature.npy")

        feature_thread = threading.Thread(target=extract_features)
        feature_thread.start()
        # feature_thread.join()
        
    def compare_audio_clip(self,audio_data):
        print("Comparing received voice clip...")
        audio_array = np.array(audio_data)

        # Normalize and save audio as WAV file
        audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        file_name = f"recordings/voice_clip_{int(time.time())}.wav"    
           
        # Create a thread for feature extraction
        def extract_features():
            print("Reducing noise and saving audio clip...")
            audio_array_reduced = nr.reduce_noise(y=audio_array, sr=44100)
            sf.write(file_name, audio_array_reduced, samplerate=44100)        
            print(f"Voice clip saved as: {file_name}")
            print("Extracting features...")
            voiced_features = feature_extraction(file_name, fs=44100)
            os.remove(file_name)
            saved_features_file = os.listdir("recordings")[0]
            model_file = os.listdir("public")[0]
            verification_result = verify_speaker(voiced_features, f"recordings/{saved_features_file}", f"public/{model_file}")
            verification_result = int(verification_result)
            self.verification_result_signal.emit(verification_result)
            self.voice_thread.logged_in=True

        feature_thread = threading.Thread(target=extract_features)
        feature_thread.start()
        # feature_thread.join()

class OpenGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.particles = [Particle() for _ in range(100)]  # Create 100 particles
        self.loudness = 0.0
        self.speech_text = ""
        self.background_color = (0.2, 0.1, 0.1, 1.0)

        # Set up a timer for animation updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)  # Call update every frame
        self.timer.start(16)  # Approximately 60 FPS

        # QLabel for displaying speech text
        self.speech_label = QLabel(self)
        self.speech_label.setGeometry(10, 10, 780, 30)
        self.speech_label.setStyleSheet("color: white; font-size: 18px;")

    def initializeGL(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(self.background_color[0], self.background_color[1], self.background_color[2], self.background_color[3])
        for particle in self.particles:
            particle.update_color((1.0, 0.0, 0.0), (1.0, 0.5, 0.5))

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClearColor(self.background_color[0], self.background_color[1], self.background_color[2], self.background_color[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw particles
        for particle in self.particles:
            particle.update(self.loudness)
            particle.draw()

        self.speech_label.setText(f"Recognized Speech: {self.speech_text}")

    def update_loudness(self, loudness):
        self.loudness = loudness

    def update_speech_text(self, text):
        self.speech_text = text
        self.update()
    
    def set_color_based_on_verification_result(self, result):
        if result == 0:
            for particle in self.particles:
                particle.update_color((0.0, 0.8, 1.0), (0.5, 0.8, 1.0))
            self.background_color = (0.1, 0.1, 0.2, 1.0)
        else:
            for particle in self.particles:
                particle.update_color((1.0, 0.0, 0.0), (1.0, 0.5, 0.5))
            self.background_color=(0.2, 0.1, 0.1, 1.0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nova")
        self.setGeometry(100, 100, 800, 600)

        # OpenGL Widget
        self.opengl_widget = OpenGLWidget()
        self.setCentralWidget(self.opengl_widget)

        self.general_functionality_thread = GeneralFunctionalityThread()
        self.general_functionality_thread.start()
        self.general_functionality_thread.verification_result_signal.connect(self.opengl_widget.set_color_based_on_verification_result)
        
        # Start real-time loudness calculation in a separate thread
        self.audio_thread = AudioThread()
        self.audio_thread.loudness_signal.connect(self.opengl_widget.update_loudness)
        self.audio_thread.start()

        # Start voice recognition in a separate thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.speech_text_signal.connect(self.opengl_widget.update_speech_text)
        self.audio_thread.audio_signal.connect(self.voice_thread.queue.put)
        self.audio_thread.loudness_signal.connect(self.voice_thread.update_loudness)
        self.voice_thread.set_general_functionality_thread(self.general_functionality_thread)
        self.voice_thread.pass_audio_signal.connect(self.general_functionality_thread.process_audio_clip)
        self.general_functionality_thread.voice_thread=self.voice_thread
        self.voice_thread.start()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())