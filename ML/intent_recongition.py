import os
import pickle
import json
from sentence_transformers import SentenceTransformer, util
import torch

base_dir = "public/"


def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def load_intents():
    if os.path.exists(base_dir + "intents.json"):
        intents = json.load(open(base_dir + "intents.json"))
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
        }
        json.dump(intents, open(base_dir + "intents.json", "w"))
    return intents


def load_intent_embeddings(model, intents=None):
    if os.path.exists(base_dir + "embeddings.pkl"):
        intent_embeddings = pickle.load(open(base_dir + "embeddings.pkl", "rb"))
    else:
        intent_embeddings = []
        for intent in list(intents.values()):
            print(intent)
            intent_embeddings.append(model.encode(intent, convert_to_tensor=True))
        intent_embeddings = torch.cat(intent_embeddings)
        pickle.dump(intent_embeddings, open(base_dir + "embeddings.pkl", "wb"))
    return intent_embeddings


def recognize_intent(
    model, user_input, intent_labels, intent_examples, intent_embeddings, threshold=0.5
):
    # Encode the user input
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(user_embedding, intent_embeddings)

    # Find the best match
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[0, best_match_idx].item()

    # Determine which intent the match belongs to
    running_total_similarity = 0.0
    matched_intent = None
    for idx, intent_label in enumerate(intent_labels):
        num_examples = len(intent_examples[intent_label])
        intent_start_idx = sum(
            len(intent_examples[label]) for label in intent_labels[:idx]
        )
        intent_end_idx = intent_start_idx + num_examples

        # If the best match index is within the range of current intent examples, update
        if intent_start_idx <= best_match_idx < intent_end_idx:
            matched_intent = intent_label
            running_total_similarity += best_match_score
            break
    if best_match_score >= threshold:
        return matched_intent, None, best_match_score
    else:
        return "Unrecognized", matched_intent, best_match_score


def fallback_to_llm(user_input):
    # Placeholder for LLM API call (Replace with actual implementation)
    print(f"Calling LLM to handle: '{user_input}'")
    # Mock LLM response (Replace with actual LLM intent detection)
    return "NewIntent", [f"User wants to do something new related to '{user_input}'"]


def add_new_intent(intents, model, new_intent_label, new_intent_description):
    if new_intent_label not in list(intents.keys()):
        # Add the new intent
        intents[new_intent_label] = new_intent_description
        json.dump(intents, open(base_dir + "intents.json", "w"))
        
        os.remove(base_dir + "embeddings.pkl")

        # Recompute embeddings
        intent_embeddings = load_intent_embeddings(model, intents)
        
        print(f"New intent '{new_intent_label}' added successfully.")
    else:
        print(f"Intent '{new_intent_label}' already exists.")


def intent_recongition(user_input, threshold=0.5):
    os.makedirs(base_dir, exist_ok=True)
    # Load the model
    model = load_model()

    # Load the intents
    intents = load_intents()

    # Load the intent embeddings
    intent_embeddings = load_intent_embeddings(model, intents)

    # Recognize the intent
    intent_label, closest_label, score = recognize_intent(
        model, user_input, list(intents.keys()), intents, intent_embeddings, threshold
    )

    if intent_label == "Unrecognized":
        print(f"Unrecognized intent. Fallback to LLM.")
        print(closest_label, score)
        # intent_label, response = fallback_to_llm(user_input)
        # add_new_intent(intents, model, intent_label, response)
    else:
        response = intents[intent_label]

    return intent_label, response, score


while True:
    user_input = input("Enter your query: ")
    if user_input == "exit":
        break
    intent_label, response, score = intent_recongition(user_input)
    print(f"Intent: {intent_label}\nResponse: {response}\nScore: {score}\n")
