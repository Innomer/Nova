from gradio_client import Client, handle_file
import os
class AudioTranscriber:
    def __init__(self):
        self.client = Client("hf-audio/whisper-large-v3")

    def transcribe_audio(self, file_path):
        result = self.client.predict(
            inputs=handle_file(file_path),
            task="transcribe",
            api_name="/predict"
        )
        os.remove(file_path)
        return result