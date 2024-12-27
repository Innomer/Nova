import os
import pickle
import json
from sentence_transformers import SentenceTransformer, util
import torch
from gemini_calls import fetch_intent_and_examples
import logging
class IntentRecognition:
    def __init__(self, base_dir="public/", threshold=0.5):
        self.base_dir = base_dir
        self.threshold = threshold
        self.model = self.load_model()
        self.intents = self.load_intents()
        self.intent_embeddings = self.load_intent_embeddings()
        self.logger = logging.getLogger(__name__)
        self.log_dir = "logs/"
        self.logger.setLevel(logging.DEBUG)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        logging.basicConfig(filename=f"{self.log_dir}/intent_recognition.log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")

    def load_model(self):
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.logger.info("Sentence Transformer loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading Sentence Transformer: {e}")
            model = None
        return model

    def load_intents(self):
        try:
            if os.path.exists(self.base_dir + "intents.json"):
                intents = json.load(open(self.base_dir + "intents.json"))
                self.logger.info("Intents loaded successfully.")
            else:
                intents = {
                    "Login": [
                        "User wants to log in or sign in to their account.",
                        "User wants use an existing account.",
                        "Log me in.",
                        "Sign me in.",
                        "Log in.",
                        "Sign in.",
                    ],
                    "Register": [
                        "User wants to register or sign up for an account.",
                        "User wants to make a new account.",
                        "Create a account.",
                        "Sign me up.",
                        "Register.",
                        "Sign up.",
                    ],
                    "Greet": [
                        "User says hello or greets.",
                        "User says hi.",
                        "User says hello.",
                        "User greets.",
                    ],
                    "Logout": [
                        "User wants to logout or lock his account.",
                        "User wants to signout.",
                        "User wants to lock his account.",
                        "Log me out.",
                        "Lock my account.",
                        "Logout.",
                        "Lock account.",
                    ],
                    "Exit": [
                        "User wants to exit or close the application.",
                        "User wants the app to shutdown.",
                        "User wants to exit.",
                        "Close the app.",
                        "Exit.",
                        "Shutdown.",
                    ],
                }
                json.dump(intents, open(self.base_dir + "intents.json", "w"))
                self.logger.info("Intents file not found. Default intents loaded.")
        except Exception as e:
            self.logger.error(f"Error loading intents: {e}")
            intents = None
        return intents

    def load_intent_embeddings(self):
        try:
            if os.path.exists(self.base_dir + "embeddings.pkl"):
                intent_embeddings = pickle.load(open(self.base_dir + "embeddings.pkl", "rb"))
                self.logger.info("Intent embeddings loaded successfully.")
            else:
                intent_embeddings = []
                for intent in list(self.intents.values()):
                    intent_embeddings.append(self.model.encode(intent, convert_to_tensor=True))
                intent_embeddings = torch.cat(intent_embeddings)
                pickle.dump(intent_embeddings, open(self.base_dir + "embeddings.pkl", "wb"))
                self.logger.info("Intent embeddings file not found. Embeddings computed and saved.")
        except Exception as e:
            self.logger.error(f"Error loading intent embeddings: {e}")
            intent_embeddings = None
        return intent_embeddings

    def recognize_intent(self, user_input):
        try:
            user_embedding = self.model.encode(user_input, convert_to_tensor=True)
            self.logger.info("User embedding computed successfully.")
            
            similarities = util.pytorch_cos_sim(user_embedding, self.intent_embeddings)

            best_match_idx = similarities.argmax().item()
            best_match_score = similarities[0, best_match_idx].item()

            running_total_similarity = 0.0
            matched_intent = None
            for idx, intent_label in enumerate(self.intents.keys()):
                num_examples = len(self.intents[intent_label])
                intent_start_idx = sum(
                    len(self.intents[label]) for label in list(self.intents.keys())[:idx]
                )
                intent_end_idx = intent_start_idx + num_examples

                if intent_start_idx <= best_match_idx < intent_end_idx:
                    matched_intent = intent_label
                    running_total_similarity += best_match_score
                    break

            if best_match_score >= self.threshold:
                self.logger.info(f"Intent recognized locally: {matched_intent}")
                return matched_intent, self.intents[matched_intent], best_match_score
            else:
                self.logger.info("Intent not recognized locally. Fetching from Gemini.")
                new_intent_label, new_intent_description = fetch_intent_and_examples(self.intents, user_input)
                if new_intent_label.lower() not in list(map(str.lower,list(self.intents.keys()))):
                    print(f"New intent being added: {new_intent_label}")
                    self.add_new_intent(new_intent_label, new_intent_description)
                return new_intent_label, new_intent_description, 1.0
        except Exception as e:
            self.logger.error(f"Error in recognizing intent: {e}")
            return None, None, None

    def add_new_intent(self, new_intent_label, new_intent_description):
        try:
            if new_intent_label not in list(self.intents.keys()):
                self.intents[new_intent_label] = new_intent_description
                json.dump(self.intents, open(self.base_dir + "intents.json", "w"))
                self.logger.debug("Intents file updated successfully.")

                os.remove(self.base_dir + "embeddings.pkl")
                self.logger.debug("Old embeddings file removed.")

                self.intent_embeddings = self.load_intent_embeddings()
                self.logger.info(f"New intent '{new_intent_label}' added successfully.")
            else:
                self.logger.info(f"Intent '{new_intent_label}' already exists.")
        except Exception as e:
            self.logger.error(f"Error in adding new intent: {e}")

    def intent_recognition(self, user_input):
        intent_label, closest_label, score = self.recognize_intent(user_input)
        response = self.intents[intent_label]
        return intent_label, response, score