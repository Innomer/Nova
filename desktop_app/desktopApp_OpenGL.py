import sys
import threading
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QPropertyAnimation, QRect, QCoreApplication, pyqtProperty
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt
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
import librosa
from audio_transcribe import AudioTranscriber
from intent_recognition import IntentRecognition
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import io
from cohere_calls import generate_cohere_response
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
        self.processing=False
        self.default_color=(0.0, 0.8, 1.0)
        self.default_trail_color=(0.5, 0.8, 1.0)
        self.processing_color=(0.0,0.6,0.6)
        self.processing_trail_color=(0.5,0.6,0.6)

    def update(self, loudness):
        if not self.processing:
            # Update radius and speed based on loudness
            self.radius = 0.1 + loudness / 2500.0
            self.speed = 0.1 + loudness / 25000.0
            self.color=self.default_color
            self.trail_color=self.default_trail_color
        else:
            self.radius = np.random.uniform(0.3, 0.8)
            self.speed = 0.5
            self.color=self.processing_color
            self.trail_color=self.processing_trail_color

        # Update angle
        self.angle += self.speed
        if self.angle >= 360:
            self.angle -= 360

        # Calculate new position
        angle_rad = radians(self.angle)
        if not self.processing:
            x = self.radius * cos(angle_rad) + pnoise2(self.x_offset, self.y_offset)
            y = self.radius * sin(angle_rad) + pnoise2(self.x_offset + 0.5, self.y_offset + 0.5)

            # Update Perlin noise offsets for smooth movement
            self.x_offset += 0.01
            self.y_offset += 0.01
        else:
            x = self.radius * cos(angle_rad)
            y = self.radius * sin(angle_rad)

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
                data = stream.read(1024, exception_on_overflow=False)

                audio_data = np.frombuffer(data, dtype=np.int16)

                loudness = np.abs(audio_data).mean()
                self.loudness_signal.emit(loudness)
                self.audio_signal.emit(audio_data)

            except Exception as e:
                print(f"Audio streaming error: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

class VoiceRecognitionThread(QThread):
    speech_text_signal = pyqtSignal(str)  # Signal to pass text to the main thread
    pass_audio_signal = pyqtSignal(np.ndarray)  # Signal to pass audio data to GeneralFunctionalityThread
    processing_signal = pyqtSignal(bool)  # Signal to indicate processing state
    exit_signal = pyqtSignal(bool)  # Signal to exit the thread

    def __init__(self):
        super().__init__()
        # self.recognizer = sr.Recognizer()
        self.recognizer=AudioTranscriber()
        self.intent_recognizer = IntentRecognition()
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
        self.registered_once = False if not os.path.exists("recordings/user_voice_features.npy") else True
        self.logged_in=False
        
        self.elevenlabs = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
        self.is_responding=False

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
                audio_data = self.queue.get()

                # Continuously add audio data to the circular buffer
                self.audio_buffer.extend(audio_data)
                self.sample_counter += len(audio_data)

                # Handle loudness-triggered audio processing
                if self.loudness >= self.activation_threshold and self.is_responding==False:
                    if not self.processing:
                        self.processing = True
                        self.loudness_start_time = time.time()
                        self.start_sample_index = self.sample_counter
                        print("Loudness above threshold, starting to track segment...")

                elif self.processing:
                    # self.processing_signal.emit(True)
                    # Stop processing if loudness is low for 5 seconds
                    if time.time() - self.loudness_start_time > 5:
                        self.processing = False
                        self.end_sample_index = self.sample_counter
                        print("Loudness below threshold, extracting audio segment...")

                        # Extract relevant portion of the audio buffer
                        margin_samples = int(self.margin_seconds * 44100)
                        segment = self.get_relevant_audio(margin_samples)
                        
                        self.start_sample_index = None
                        self.end_sample_index = None
                        self.audio_buffer.clear()

                        # Plot, play, and recognize speech from the segment
                        # self.plot_audio_buffer(segment)
                        # self.play_audio(segment)
                        self.recognize_speech(segment)
                # self.processing_signal.emit(False)

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
        self.ax.clear()
        self.ax.plot(audio_array, lw=1, color='blue')
        self.ax.set_title("Audio Segment Data")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def play_response_audio(self, text):
        self.is_responding=True
        response = self.elevenlabs.generate(text=text, voice="Rachel", model="eleven_flash_v2_5")
        save(response, 'recordings/temp_output.mp3')
        audio=AudioSegment.from_mp3('recordings/temp_output.mp3')
        silence = AudioSegment.silent(duration=1500)
        audio = silence + audio + silence
        play(audio)
        os.remove("recordings/temp_output.mp3")
        self.is_responding=False
        
    def get_and_play_responses(self, intent_label,text,tense, response_text=None):
        if response_text is None:
            text_generic_response=generate_cohere_response(intent_label, text, tense)
            self.play_response_audio(text_generic_response)
        else:
            self.play_response_audio(response_text)
        
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
            self.processing_signal.emit(True)
            print("Processing speech recognition...")
            audio_bytes = audio_array.tobytes()
            audio = sr.AudioData(audio_bytes, 44100, 2)
            sf.write("recordings/temp_voice_clip.wav", audio_array, 44100, subtype='PCM_16')
            text=self.recognizer.transcribe_audio("recordings/temp_voice_clip.wav")
            # text = self.recognizer.recognize_google(audio)
            print(f"Recognized Speech: {text}")

            self.speech_text_signal.emit(text)
            
            intent_label, response, score = self.intent_recognizer.intent_recognition(text)
            
            self.intent = intent_label.lower()
            print(f"Intent: {self.intent}")
            if self.intent=="register" and not self.registered_once:
                response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1))
                response_thread.start()
                response_thread.join()
                self.registered_once = True
                self.pass_audio_signal.emit(audio_array)
            elif self.intent=='register' and self.registered_once:
                if self.logged_in:
                    self.intent = "register"
                    response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1))
                    response_thread.start()
                    response_thread.join()
                    self.pass_audio_signal.emit(audio_array)
                else:
                    self.speech_text_signal.emit("Restricted Access! Please Login!") # Remind user to login
                    response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1, "Restricted Access! Please Login!"))
                    response_thread.start()
                    response_thread.join()
            elif self.intent=='login':
                self.speech_text_signal.emit("Logging in...")
                response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1))
                response_thread.start()
                response_thread.join()
                self.pass_audio_signal.emit(audio_array)
            elif self.intent=='logout':
                self.speech_text_signal.emit("Logging out...")
                response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1))
                response_thread.start()    
                response_thread.join()
                self.pass_audio_signal.emit(audio_array)
                self.logged_in=False
            elif self.intent=='exit':
                response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1, "I will be shutting down now! Goodbye!"))
                response_thread.start()
                response_thread.join()
                self.exit_signal.emit(True)
            elif self.intent=='greet':
                response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,0))
                response_thread.start()
                response_thread.join()
            else:
                if self.logged_in:
                    print(self.intent , response)
                    response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,0))
                    response_thread.start()
                    response_thread.join()
                else:
                    self.speech_text_signal.emit("Restricted Access! Please Login!")
                    response_thread=threading.Thread(target=self.get_and_play_responses, args=(intent_label,text,1, "Restricted Access! Please Login!"))
                    response_thread.start()
                    response_thread.join()
            self.processing_signal.emit(False)
        except Exception as e:
            response_thread=threading.Thread(target=self.get_and_play_responses, args=("error","error",1, "I am sorry, I did not understand that!"))
            response_thread.start()
            response_thread.join()
            print(f"Speech recognition error: {e}")

class GeneralFunctionalityThread(QThread):
    verification_result_signal = pyqtSignal(int)  # Signal to pass verification result to the main thread
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.voice_thread=None
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("public", exist_ok=True)

    def run(self):
        """Thread entry point, runs continuously."""
        pass

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
            audio, fs = librosa.load(file_name, sr=44100)
            resampled_audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
            sf.write(file_name, resampled_audio, 16000, subtype='PCM_16')
            print(f"Voice clip saved as: {file_name}")
            print("Extracting features...")
            voiced_features = feature_extraction(file_name)
            with open("recordings/user_voice_features.npy", "wb") as f:
                np.save(f, voiced_features)
            os.remove(file_name)
            print("Voice features saved as: user_voice_feature.npy")

        feature_thread = threading.Thread(target=extract_features)
        feature_thread.start()
        
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
            audio, fs = librosa.load(file_name, sr=44100)
            resampled_audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
            sf.write(file_name, resampled_audio, 16000, subtype='PCM_16')   
            print(f"Voice clip saved as: {file_name}")
            print("Extracting features...")
            voiced_features = feature_extraction(file_name)
            os.remove(file_name)
            saved_features_file = "user_voice_features.npy"
            # model_file = os.listdir("public")[0]
            model_file = 'siamese_model.h5'
            verification_result = verify_speaker(voiced_features, f"recordings/{saved_features_file}", f"public/{model_file}")
            verification_result = int(verification_result)
            self.verification_result_signal.emit(verification_result)
            print(f"Verification result: {verification_result}")
            self.voice_thread.logged_in=True if verification_result==0 else False
            if verification_result==1:
                inner_thread=threading.Thread(target=self.voice_thread.get_and_play_responses, args=("error","error",1, "Voice Sample did not match! Access Denied!"))
                inner_thread.start()
                inner_thread.join()

        feature_thread = threading.Thread(target=extract_features)
        feature_thread.start()

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
    
    def set_processing_state(self, state):
        for particle in self.particles:
            particle.processing=state
    
    def set_color_based_on_verification_result(self, result):
        if result == 0:
            for particle in self.particles:
                particle.update_color((0.0, 0.8, 1.0), (0.5, 0.8, 1.0))
            self.background_color = (0.1, 0.1, 0.2, 1.0)
        else:
            for particle in self.particles:
                particle.update_color((1.0, 0.0, 0.0), (1.0, 0.5, 0.5))
            self.background_color=(0.2, 0.1, 0.1, 1.0)

class CanvasOverlay(QWidget):
    def __init__(self, parent=None):
        super(CanvasOverlay, self).__init__(parent)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.components = []  # Track runtime components
        self._opacity = 0.0  # Start with 0 opacity (hidden)
        self.has_background = False

        # Set up animations
        self.fade_animation = QPropertyAnimation(self, b"opacity")
        self.fade_animation.setDuration(500)  # Duration in milliseconds

        self.resize_animation = QPropertyAnimation(self, b"geometry")
        self.resize_animation.setDuration(500)  # Duration in milliseconds

        self.setStyleSheet("background: transparent;")

    @pyqtProperty(float)
    def opacity(self):
        """Custom property for animating opacity."""
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self._opacity = value
        self.setWindowOpacity(self._opacity)  # Adjust the window opacity
        self.update()

    def add_component(self, component):
        """Dynamically add a component to the canvas."""
        self.components.append(component)
        component.setParent(self)
        component.show()
        self.show_with_animation()

    def clear_components(self):
        """Remove all components from the canvas."""
        for component in self.components:
            component.setParent(None)
            component.deleteLater()
        self.components = []
        self.hide_with_animation()

    def show_with_animation(self):
        """Smoothly fade in and grow the canvas."""
        if not self.has_background:  # Only animate if not already visible
            self.has_background = True
            self.show()  # Make sure the widget is visible

            # Set up fade-in animation
            self.fade_animation.stop()
            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(1.0)

            # Set up grow animation
            self.resize_animation.stop()
            parent_rect = self.parentWidget().rect()
            start_rect = QRect(parent_rect.width() // 2, parent_rect.height() // 2, 0, 0)
            end_rect = parent_rect.adjusted(20, 20, -20, -20)  # Slightly smaller than parent

            self.setGeometry(start_rect)  # Start with collapsed size
            self.resize_animation.setStartValue(start_rect)
            self.resize_animation.setEndValue(end_rect)

            # Start animations
            self.fade_animation.start()
            self.resize_animation.start()

    def hide_with_animation(self):
        """Smoothly fade out and shrink the canvas."""
        if self.has_background:  # Only animate if currently visible
            
            # Set up fade-out animation
            self.fade_animation.stop()
            self.fade_animation.setStartValue(1.0)
            self.fade_animation.setEndValue(0.0)

            # Set up shrink animation
            self.resize_animation.stop()
            parent_rect = self.parentWidget().rect()
            end_rect = QRect(parent_rect.width() // 2, parent_rect.height() // 2, 0, 0)

            self.resize_animation.setStartValue(self.parentWidget().rect().adjusted(20, 20, -20, -20))
            self.resize_animation.setEndValue(end_rect)

            # Start animations
            self.fade_animation.start()
            self.resize_animation.start()

            # Hide widget when animation finishes
            self.fade_animation.finished.connect(self._finalize_hide)

    def _finalize_hide(self):
        """Finalize hiding the canvas."""
        self.has_background = False
        self.hide()

    def paintEvent(self, event):
        """Draw the semi-transparent background if needed."""
        if self.has_background:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Semi-transparent rounded rectangle
            rect = self.rect().adjusted(0, 0, 0, 0)
            color = QColor(0, 0, 0, 128)  # Semi-transparent black
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 15, 15)  # Rounded corners

    def mousePressEvent(self, event):
        """Pass mouse press events to child widgets."""
        for component in self.components:
            if component.geometry().contains(event.pos()):
                QCoreApplication.sendEvent(component, event)

    def mouseReleaseEvent(self, event):
        """Pass mouse release events to child widgets."""
        for component in self.components:
            if component.geometry().contains(event.pos()):
                QCoreApplication.sendEvent(component, event)

    def mouseMoveEvent(self, event):
        """Pass mouse move events to child widgets."""
        for component in self.components:
            if component.geometry().contains(event.pos()):
                QCoreApplication.sendEvent(component, event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nova")
        self.setGeometry(100, 100, 800, 600)
        self.threads = []

        # OpenGL Widget
        self.opengl_widget = OpenGLWidget()
        self.setCentralWidget(self.opengl_widget)
        
        # Add Canvas Overlay
        self.canvas_overlay = CanvasOverlay(self)
        self.canvas_overlay.setGeometry(self.opengl_widget.geometry())

        self.general_functionality_thread = GeneralFunctionalityThread()
        self.general_functionality_thread.start()
        self.general_functionality_thread.verification_result_signal.connect(self.opengl_widget.set_color_based_on_verification_result)
        self.threads.append(self.general_functionality_thread)
        
        # Start real-time loudness calculation in a separate thread
        self.audio_thread = AudioThread()
        self.audio_thread.loudness_signal.connect(self.opengl_widget.update_loudness)
        self.audio_thread.start()
        self.threads.append(self.audio_thread)

        # Start voice recognition in a separate thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.speech_text_signal.connect(self.opengl_widget.update_speech_text)
        self.audio_thread.audio_signal.connect(self.voice_thread.queue.put)
        self.audio_thread.loudness_signal.connect(self.voice_thread.update_loudness)
        self.voice_thread.set_general_functionality_thread(self.general_functionality_thread)
        self.voice_thread.pass_audio_signal.connect(self.general_functionality_thread.process_audio_clip)
        self.general_functionality_thread.voice_thread=self.voice_thread
        self.voice_thread.processing_signal.connect(self.opengl_widget.set_processing_state)
        self.voice_thread.exit_signal.connect(self.closeEvent)
        self.voice_thread.start()
        self.threads.append(self.voice_thread)
    
    # def add_sample_components_to_canvas(self):
    #     """Example to dynamically add components to the canvas."""
    #     from PyQt5.QtWidgets import QPushButton, QLabel

    #     # Example button
    #     button = QPushButton("Click Me", self)
    #     button.move(100, 100)
    #     button.clicked.connect(lambda: print("Button clicked!"))
    #     self.canvas_overlay.add_component(button)

    #     # Example label
    #     label = QLabel("Dynamic Overlay Label", self)
    #     label.move(100, 150)
    #     label.setStyleSheet("color: white; font-size: 14px;")
    #     self.canvas_overlay.add_component(label)

    #     # Example: Clear components after 5 seconds (for demonstration)
    #     QTimer.singleShot(20000, self.canvas_overlay.clear_components)
        
    def resizeEvent(self, event):
        """Ensure the canvas overlay matches the OpenGL widget size on resize."""
        super(MainWindow, self).resizeEvent(event)
        self.canvas_overlay.setGeometry(self.opengl_widget.geometry())
        self.canvas_overlay.setGeometry(self.canvas_overlay.parentWidget().rect().adjusted(20, 20, -20, -20))
    
    def close(self,event=True):
        if event:
            for thread in self.threads:
                thread.quit()
                thread.wait()
        
if __name__ == "__main__":
    load_dotenv()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())