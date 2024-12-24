import pygame
import random
import math
import pyaudio
import numpy as np
import os
import time
import wave
import threading
import speech_recognition as sr
import pathlib
import textwrap, requests
# import google.generativeai as genai
from github_commands import GitHubClient


pygame.init()
# genai.configure(api_key="")
# model = genai.GenerativeModel('gemini-pro')
# gcC= GitHubClient(username="Innomer", password="Innomer@123")

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Sound Animation")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PARTICLE_COUNT = 50
RECORDING_DIR = "recordings"
IS_RECORDING = False

if not os.path.exists(RECORDING_DIR):
    os.makedirs(RECORDING_DIR)


class Thread(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(self.image, WHITE, (5, 5), 5)  # Draw a circle on the surface
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = random.randint(0, 360)
        self.radius = random.randint(50, 200)
        self.min_radius = 5  # Minimum radius of rotation
        self.max_radius = 50  # Maximum radius of rotation
        self.speed = 1  # Adjust the speed here
        self.default_change_speed = 2
        self.radius_change_speed = self.default_change_speed  # Adjust the speed at which the radius changes
        self.noise_range = 5  # Adjust the noise range here
        self.noise_speed_x = 0.01  # Adjust the noise speed for x-coordinate here
        self.noise_speed_y = 0.01  # Adjust the noise speed for y-coordinate here
        self.noise_offset_x = random.uniform(0, 1000)  # Randomize the noise offset for x-coordinate
        self.noise_offset_y = random.uniform(0, 1000)  # Randomize the noise offset for y-coordinate
        self.default_radius_noise = 2000
        self.radius_noise = self.default_radius_noise  # Maximum amount of noise to add to radius
        self.trail_length = 10  # Length of the trail
        self.trail = []  # List to store previous positions

    def update(self, sound_level):
        self.angle += self.speed
        noise_x = self.noise_range * math.cos(self.noise_offset_x + self.angle * self.noise_speed_x)  # Generate smooth noise for x-coordinate
        noise_y = self.noise_range * math.sin(self.noise_offset_y + self.angle * self.noise_speed_y)  # Generate smooth noise for y-coordinate

        # Adjust radius based on sound level
        base_radius = random.uniform(self.min_radius, self.max_radius)
        noise_radius = random.uniform(-self.radius_noise, self.radius_noise)
        target_radius = base_radius + noise_radius + sound_level
        if target_radius > self.radius:
            self.radius += self.radius_change_speed
        elif target_radius < self.radius:
            self.radius -= self.radius_change_speed

        self.rect.x = width // 2 + self.radius * math.cos(math.radians(self.angle)) + noise_x
        self.rect.y = height // 2 + self.radius * math.sin(math.radians(self.angle)) + noise_y

        # Update trail
        self.trail.append((self.rect.x, self.rect.y))
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)

        if sound_level < 200:
            self.radius_change_speed = 10
            self.radius_noise = 1
        elif sound_level < 500:
            self.radius_change_speed = 100
            self.radius_noise = 0.1
        else:
            self.radius_change_speed = self.default_change_speed

        # Adjust speed based on sound level
        self.speed = 0.1 + sound_level / 500
        if self.speed < 2:
            self.speed = 2

    def draw_trail(self, screen):
        for i in range(1, len(self.trail)):
            pygame.draw.line(screen, WHITE, self.trail[i - 1], self.trail[i], 2)


all_threads = pygame.sprite.Group()

for _ in range(PARTICLE_COUNT):
    thread = Thread(width // 2, height // 2)
    all_threads.add(thread)

# Initialize audio stream
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()
stream = None
text = ""


# Test microphone
def test_microphone():
    global stream
    print("Testing microphone...")
    try:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        device_info = p.get_default_input_device_info()
        device_index = device_info['index']
        device_name = p.get_device_info_by_index(device_index)['name']
        print("Using input device:", device_name)
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index)
        data = stream.read(CHUNK)
        print("Microphone is working.")
    except Exception as e:
        print("Microphone is not working. Error:", str(e))

STOP=False
# Function to record audio and save to file
def record_audio(stop_event):
    global text
    global IS_RECORDING
    global STOP
    if IS_RECORDING:
        text = "Already Speaking. Please Stay Silent for 5 seconds to end"
        STOP=True
        return
    IS_RECORDING = True
    text="Speaking"
    frames = []
    start_time = time.time()
    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)
        audio = np.frombuffer(data, dtype=np.int16)
        sound_level = np.abs(audio).mean()
        if sound_level < 100:
            if time.time() - start_time > 5 or STOP:
                print("Sound level below 100 for more than 5 seconds. Stopping recording.")
                IS_RECORDING = False
                STOP=False
                break
        else:
            start_time = time.time()
        time.sleep(0.01)  # Add a small delay to prevent high CPU usage

    # Save recorded audio to file
    filename = os.path.join(RECORDING_DIR, f"record_{int(time.time())}.wav")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved as {filename}")

    # Send recorded audio to API
    url = "https://d56b-2409-40c0-1048-fe65-597d-2429-50c1-3560.ngrok-free.app/file"
    files = {'file_audio': open(filename, 'rb')}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        # Receive and process API response
        api_response = response.json()
        lst=api_response
        print(api_response)
        action=api_response[0]
        file_name=api_response[1]
        print("Action:", action)
        print("File Name:", file_name)

        # Search for the file in the system
        file_paths = search_file(file_name)
        print("File Paths:", file_paths)
        # Handle the file paths as needed
    IS_RECORDING = False

    # Perform speech recognition
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(filename) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text2 = recognizer.recognize_google(audio_data)
#             print("Speech Recognition Result:", text2)
#             action=model.generate_content(f"Can you extract the action involved in the following command? \n {text2}").text
#             entities=model.generate_content(f"Can you extract the entities involved in the following command? \n {text2}").text
            
#             # task, action, entities = extract_task_action_entities(text)
#             print(f"Task: {task}")
#             print(f"Action: {action}")
#             print("Entities:")
#             for key, value in entities.items():
#                 print(f"{key}: {value}")
#             if action == "Upload":
#                     for i in entities.values():
#                         file_paths = search_file(entities[i])
#                         if file_paths:
#                             print(f"File found at: {file_paths[0]}")
#                         else:
#                             print(f"File not found: {entities[i]}")
#                         repo=gcC.get_repository("Innomer/Portfolio-Website")
#                         gcC.commit_changes(repo, "main", file_paths, "Added a new recording")
#                         text="File Uploaded"
#                     print("DONE")
#             if action == "Download":
#                 repo=gcC.get_repository("Innomer/Portfolio-Website")
#                 gcC.pull_changes(repo, "main")

#             if action == "Commit":
#                 repo=gcC.get_repository("Innomer/Portfolio-Website")
#                 gcC.get_commits(repo, "main")
#             if action == "Repository":
#                 repo=gcC.get_repository()
#         except sr.UnknownValueError:
#             print("Speech Recognition could not understand audio")
#         except sr.RequestError as e:
#             print("Could not request results from Google Speech Recognition service; {0}".format(e))

# import re

# def extract_task_action_entities(input_string):
#     # Define regular expressions for Task, Action, and Entities
#     task_pattern = r"\*\*Task\*\*: (.+)"
#     action_pattern = r"\*\*Action\*\*: (.+)"
#     entity_pattern = r"\*\*Entities\*\*:(.+)"

#     # Search for matches using regex
#     task_match = re.search(task_pattern, input_string)
#     action_match = re.search(action_pattern, input_string)
#     entity_match = re.search(entity_pattern, input_string)

#     # Extract Task, Action, and Entities from matches
#     task = task_match.group(1).strip() if task_match else None
#     action = action_match.group(1).strip() if action_match else None
#     entities = {}

#     if entity_match:
#         entity_text = entity_match.group(1)
#         # Extract entities using regex
#         entity_pairs = re.findall(r"- (\w+):\s*([\w.]+)", entity_text)
#         entities = {key: value for key, value in entity_pairs}

#     return task, action, entities

# def search_file(file_name):
#     # Initialize a list to store the absolute paths of the file
#     file_paths = []

#     # Walk through the file system starting from the root directory
#     for root_dir, _, files in os.walk('/'):
#         # Check if the file exists in the current directory
#         if file_name in files:
#             # If the file exists, get its absolute path and add it to the list
#             file_path = os.path.join(root_dir, file_name)
#             file_paths.append(file_path)

#     return file_paths
test_microphone()

def search_file(file_name):
    # Initialize a list to store the absolute paths of the file
    file_paths = []

    # Walk through the file system starting from the root directory
    for root_dir, _, files in os.walk('/'):
        # Check if the file exists in the current directory
        if file_name in files:
            # If the file exists, get its absolute path and add it to the list
            file_path = os.path.join(root_dir, file_name)
            file_paths.append(file_path)

    return file_paths


def record_thread(stop_event):
    record_audio(stop_event)


def draw_ui():
    global button_rect
    button_rect = pygame.Rect(20, 40, 100, 30)
    button_text = pygame.font.SysFont(None, 20).render("Test Mic", True, BLACK)

    global record_button_rect, IS_RECORDING
    # Button to record audio
    record_button_rect = pygame.Rect(width/2-80, height-60, 150, 30)
    if IS_RECORDING:
        color="RED"
    else:
        color="WHITE"
    record_button_text = pygame.font.SysFont(None, 20).render("Speak", True, BLACK)

    font = pygame.font.SysFont(None, 30)

    # Draw button
    pygame.draw.rect(screen, WHITE, button_rect)
    # screen.blit(button_text, button_rect.topleft)
    screen.blit(button_text, (button_rect.x + button_rect.width // 2 - button_text.get_width() // 2,
                              button_rect.y + button_rect.height // 2 - button_text.get_height() // 2))


    # Draw record button
    pygame.draw.rect(screen, color, record_button_rect)
    # screen.blit(record_button_text, record_button_rect.topleft)
    screen.blit(record_button_text, (record_button_rect.x + record_button_rect.width // 2 - record_button_text.get_width() // 2,
                              record_button_rect.y + record_button_rect.height // 2 - record_button_text.get_height() // 2))

    # Display sound level
    sound_level_text = font.render(f"Sound Level: {sound_level:.2f}", True, WHITE)
    screen.blit(sound_level_text, (10, height - 40))

    sr_text = font.render(f"{text}", True, WHITE)
    screen.blit(sr_text, (width // 2 - sr_text.get_width() // 2, height // 2 - sr_text.get_height() // 2))


# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                test_microphone()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and button_rect.collidepoint(event.pos):
                test_microphone()
            elif event.button == 1 and record_button_rect.collidepoint(event.pos):
                # Start recording in a separate thread
                stop_event = threading.Event()
                record_thread_instance = threading.Thread(target=record_thread, args=(stop_event,))
                record_thread_instance.start()

    # Read audio from microphone
    if stream is not None:
        data = stream.read(CHUNK)
        audio = np.frombuffer(data, dtype=np.int16)

        sound_level = np.abs(audio).mean()

        all_threads.update(sound_level)

    screen.fill(BLACK)

    draw_ui()

    # Draw threads
    for thread in all_threads:
        screen.blit(thread.image, thread.rect)
        thread.draw_trail(screen)

    # Update the display
    pygame.display.flip()

# Stop audio stream
if stream is not None:
    stream.stop_stream()
    stream.close()
p.terminate()

# Quit Pygame
pygame.quit()
