import os
import google.generativeai as genai
from google.generativeai import caching
import datetime
import time
import dotenv
from dotenv import load_dotenv
import typing_extensions as typing
import json
import ast
import logging

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
os.makedirs("logs", exist_ok=True)
log_dir = "logs"
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(
    filename=f"{log_dir}/gemini_call.log", level=logging.INFO, format=FORMAT
)
logger.setLevel(logging.INFO)
token_count = os.environ.get("TOKEN_COUNT", 0)
num_requests = os.environ.get("NUM_REQUESTS", 0)


class Intent(typing.TypedDict):
    intent_label: str


class IntentExamples(typing.TypedDict):
    examples: list[str]

def get_gemini_model(intents, get_label=True, model_name="models/gemini-1.5-pro-latest"):
    if intents is None:
        logger.error("Intents are None")
        return None
    logger.info(f"Getting Gemini model with intents set to {get_label}")
    try:
        if get_label:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=(
                    f"You are an assistant tasked with determining the intent of the user's query. You are provided with a list of intents and examples {intents}. You should return the most suitable intent of the user as a single key with Verb Following the Noun Format (like OrderPizza). If the intent is not present in the provided list, return what you think the Intent is (as generic as possible). For example: \n Label: 'OrderPizza' \n Examples: ['I would like to order a pizza', 'Can I get a pizza', 'I want a pizza']"
                ),
            )
        else:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=(
                    f"You have to return a list of 10 examples of the user's Intent. Cover as many variations as possible."
                ),
            )
        logger.info(f"Model with intents set to {get_label} created")
        return model
    except Exception as e:
        logger.error(f"Error in getting Gemini model: {e}")
        return None


def get_gemini_response(model, prompt, get_intent=True):
    if model is None:
        logger.error("Model is None")
        return None
    if prompt is None:
        logger.error("Prompt is None")
        return None
    logger.info(
        f"Getting Gemini response with prompt: {prompt} and intent set to {get_intent}"
    )
    try:
        if get_intent:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=Intent
                ),
            )
        else:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=IntentExamples,
                ),
            )
        logger.info(f"Response received from Gemini: {response}")
        response = response.candidates[0].content.parts[0].text
        response = ast.literal_eval(response)
        return response
    except genai.exceptions.ResourceExhausted as e:
        logger.debug(f"Resource exhausted: {e}")
        return 403
    except Exception as e:
        logger.error(f"Error in getting Gemini response: {e}")
        return None


def fetch_intent_and_examples(intents, prompt):
    if intents is None:
        logger.error("Intents are None")
        return None, None
    if prompt is None:
        logger.error("Prompt is None")
        return None, None

    model = get_gemini_model(intents)
    if model is None:
        logger.error("Model is None")
        return None, None

    response = get_gemini_response(model, f"Get me intent of user query: {prompt}")
    if response == 403:
        model = get_gemini_model(intents, get_label=True, model_name="models/gemini-1.5-flash-8b-latest")
        response = get_gemini_response(model, f"Get me intent of user query: {prompt}")
    num_requests += 1
    token_count += model.count_tokens(f"Get me intent of user query: {prompt}")
    token_count += model.count_tokens(
        f"You are an assistant tasked with determining the intent of the user's query. You are provided with a list of intents and examples {intents}. You should return the most suitable intent of the user as a single key with Verb Following the Noun Format (like OrderPizza). If the intent is not present in the provided list, return what you think the Intent is (as generic as possible). For example: \n Label: 'OrderPizza' \n Examples: ['I would like to order a pizza', 'Can I get a pizza', 'I want a pizza']"
    )
    if response is None:
        logger.error("Response is None")
        return None, None

    label = None
    examples = None
    try:
        if "intent_label" in list(response.keys()):
            label = response["intent_label"]
            if label in intents.keys():
                pass
            else:
                model = get_gemini_model(intents, get_label=False)
                num_requests += 1
                token_count += model.count_tokens(
                    f"Get me examples of user query with the Intent: {label}"
                )
                token_count += model.count_tokens( 
                    f"You have to return a list of 10 examples of the user's Intent. Cover as many variations as possible."
                )
                example_response = get_gemini_response(
                    model,
                    f"Get me examples of user query with the Intent: {label}",
                    get_intent=False,
                )
                if example_response==403:
                    model = get_gemini_model(intents, get_label=False, model_name="models/gemini-1.5-flash-8b-latest")
                    example_response = get_gemini_response(
                        model,
                        f"Get me examples of user query with the Intent: {label}",
                        get_intent=False,
                    )
                if "examples" in list(example_response.keys()):
                    examples = example_response["examples"]

        logger.info(f"Intent: {label}, Examples: {examples}")
        dotenv.set_key(".env", "TOKEN_COUNT", token_count)
        dotenv.set_key(".env", "NUM_REQUESTS", num_requests)
        os.environ["TOKEN_COUNT"] = token_count
        os.environ["NUM_REQUESTS"] = num_requests
        logger.info(f"Token count: {token_count}, Number of requests: {num_requests}")
        return label, examples
    except Exception as e:
        logger.error(f"Error in fetching intent and examples: {e}")
        return None, None