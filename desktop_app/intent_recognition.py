import os
import pickle
import json
from sentence_transformers import SentenceTransformer, util
import torch

class IntentRecognition:
    def __init__(self, base_dir="public/", threshold=0.5):
        self.base_dir = base_dir
        self.threshold = threshold
        self.model = self.load_model()
        self.intents = self.load_intents()
        self.intent_embeddings = self.load_intent_embeddings()

    def load_model(self):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model

    def load_intents(self):
        if os.path.exists(self.base_dir + "intents.json"):
            intents = json.load(open(self.base_dir + "intents.json"))
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
        return intents

    def load_intent_embeddings(self):
        if os.path.exists(self.base_dir + "embeddings.pkl"):
            intent_embeddings = pickle.load(open(self.base_dir + "embeddings.pkl", "rb"))
        else:
            intent_embeddings = []
            for intent in list(self.intents.values()):
                intent_embeddings.append(self.model.encode(intent, convert_to_tensor=True))
            intent_embeddings = torch.cat(intent_embeddings)
            pickle.dump(intent_embeddings, open(self.base_dir + "embeddings.pkl", "wb"))
        return intent_embeddings

    def recognize_intent(self, user_input):
        # Encode the user input
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)

        # Compute cosine similarity
        similarities = util.pytorch_cos_sim(user_embedding, self.intent_embeddings)

        # Find the best match
        best_match_idx = similarities.argmax().item()
        best_match_score = similarities[0, best_match_idx].item()

        # Determine which intent the match belongs to
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
            return matched_intent, None, best_match_score
        else:
            return "Unrecognized", matched_intent, best_match_score

    def fallback_to_llm(self, user_input):
        # Placeholder for LLM API call (Replace with actual implementation)
        print(f"Calling LLM to handle: '{user_input}'")
        # Mock LLM response (Replace with actual LLM intent detection)
        return "NewIntent", [f"User wants to do something new related to '{user_input}'"]

    def add_new_intent(self, new_intent_label, new_intent_description):
        if new_intent_label not in list(self.intents.keys()):
            # Add the new intent
            self.intents[new_intent_label] = new_intent_description
            json.dump(self.intents, open(self.base_dir + "intents.json", "w"))

            os.remove(self.base_dir + "embeddings.pkl")

            # Recompute embeddings
            self.intent_embeddings = self.load_intent_embeddings()

            print(f"New intent '{new_intent_label}' added successfully.")
        else:
            print(f"Intent '{new_intent_label}' already exists.")

    def intent_recognition(self, user_input):
        intent_label, closest_label, score = self.recognize_intent(user_input)

        if intent_label == "Unrecognized":
            print(f"Unrecognized intent. Fallback to LLM.")
            print(closest_label, score)
            # intent_label, response = self.fallback_to_llm(user_input)
            # self.add_new_intent(intent_label, response)
        else:
            response = self.intents[intent_label]

        return intent_label, response, score

# Example usage:
# if __name__ == "__main__":
#     ir_system = IntentRecognition()

#     user_input = "I want to sign up"
#     intent_label, response, score = ir_system.intent_recognition(user_input)

#     print(f"Intent: {intent_label}")
#     print(f"Response: {response}")
#     print(f"Score: {score}")