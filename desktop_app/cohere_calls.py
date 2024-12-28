import cohere
from dotenv import load_dotenv
import os
import logging

os.makedirs("logs", exist_ok=True)
log_dir = 'logs'
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(filename=f"{log_dir}/cohere_calls.log", level=logging.INFO, format=FORMAT)
logger.setLevel(logging.INFO)

def generate_cohere_response(intent_label, user_input, tense=1):
    load_dotenv()
    try:
        co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
        if tense==1:
            prompt=f"You are an AI Voice Assistant. The user's intent is to \"{intent_label}\" and the user's input is \"{user_input}\". Provide an appropriate short and sweet response as a filler so that the user waits until the processing is complete."
        else:
            prompt=f"You are an AI Voice Assistant. The user's intent was to \"{intent_label}\" and the user's input was \"{user_input}\". Provide an appropriate short and sweet response in a sentence or two maximum."
        logger.info(f"Generating C4AI response for promt: {prompt}")
        response = co.generate(
            prompt=prompt,
            model="c4ai-aya-expanse-8b",
            truncate="END",
            presence_penalty=0.25
        )
        logger.info(f"Cohere response generated: {response.generations[0].text}")
        return response.generations[0].text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return 403