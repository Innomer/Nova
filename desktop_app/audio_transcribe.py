from gradio_client import Client, handle_file
import os
import logging
class AudioTranscriber:
    def __init__(self):
        self.client = Client("hf-audio/whisper-large-v3")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = "logs/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        logging.basicConfig(filename=f"{self.log_dir}/audio_transcribe.log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")

    def transcribe_audio(self, file_path):
        try:
            result = self.client.predict(
                inputs=handle_file(file_path),
                task="transcribe",
                api_name="/predict"
            )
            self.logger.info(f"Transcription result: {result}")
            os.remove(file_path)
            self.logger.debug(f"Deleted file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            result = "Error"
        return result