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

class Particle:
    def __init__(self):
        self.angle = np.random.uniform(0, 360)
        self.radius = np.random.uniform(0.3, 0.8)
        self.speed = np.random.uniform(0.1, 1.0)
        self.history = deque(maxlen=10)  # Store positions for trails

        self.x_offset = np.random.uniform(0, 0.5)
        self.y_offset = np.random.uniform(0, 0.5)

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

    def draw(self):
        # Draw trails
        glBegin(GL_LINE_STRIP)
        for i, (x, y) in enumerate(self.history):
            alpha = (i + 1) / len(self.history)  # Gradual fade
            glColor4f(0.5, 0.8, 1.0, alpha)  # Particle trail color
            glVertex2f(x, y)
        glEnd()

        # Draw the particle
        glColor3f(0.0, 0.8, 1.0)
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

        # Set up a live plot for dequeued audio data
        self.fig, self.ax = plt.subplots()
        self.plot_line, = self.ax.plot([], [], lw=1)
        plt.ion()
        plt.show()
        
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
                        self.plot_audio_buffer(segment)
                        self.play_audio(segment)
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

    def recognize_speech(self, audio_array):
        """Recognize speech from the extracted audio."""
        try:
            print("Processing speech recognition...")
            audio_bytes = audio_array.tobytes()
            audio = sr.AudioData(audio_bytes, 44100, 2)
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized Speech: {text}")

            self.speech_text_signal.emit(text)

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


class OpenGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.particles = [Particle() for _ in range(100)]  # Create 100 particles
        self.loudness = 0.0
        self.speech_text = ""

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
        glClearColor(0.1, 0.1, 0.2, 1.0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nova")
        self.setGeometry(100, 100, 800, 600)

        # OpenGL Widget
        self.opengl_widget = OpenGLWidget()
        self.setCentralWidget(self.opengl_widget)

        # Start real-time loudness calculation in a separate thread
        self.audio_thread = AudioThread()
        self.audio_thread.loudness_signal.connect(self.opengl_widget.update_loudness)
        self.audio_thread.start()

        # Start voice recognition in a separate thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.speech_text_signal.connect(self.opengl_widget.update_speech_text)
        self.audio_thread.audio_signal.connect(self.voice_thread.queue.put)
        self.audio_thread.loudness_signal.connect(self.voice_thread.update_loudness)
        self.voice_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())