import random
from src.preprocessing import clean_text, load_intents, extract_dataset

class DialogflowSimulatorClient:
    """
    Local Dialogflow-style baseline using rule-based pattern-match scoring.

    This class does not call Google Dialogflow.  ``training_patterns`` and
    ``training_tags`` allow evaluation to restrict the lookup table and Jaccard
    search to the training partition, preventing held-out phrase leakage.  The
    original ``DialogflowSimulatorClient(dataset_path)`` call remains supported
    for the local full-dataset demo.
    """
    def __init__(self, dataset_path: str, training_patterns=None, training_tags=None):
        self.intents_data = load_intents(dataset_path)
        self.patterns, self.tags, self.responses_dict = extract_dataset(self.intents_data)

        if (training_patterns is None) != (training_tags is None):
            raise ValueError("training_patterns and training_tags must be provided together")

        if training_patterns is None:
            examples = list(zip(self.patterns, self.tags))
        else:
            if len(training_patterns) != len(training_tags):
                raise ValueError("training_patterns and training_tags must have equal lengths")
            examples = []
            for pattern, tag in zip(training_patterns, training_tags):
                cleaned = clean_text(pattern)
                if cleaned:
                    examples.append((cleaned, tag))

        if not examples:
            raise ValueError("at least one training example is required")

        self.training_examples = examples
        self.patterns_dict = {}
        for pattern, tag in self.training_examples:
            self.patterns_dict[pattern] = tag

    @property
    def training_example_count(self) -> int:
        return len(self.training_examples)

    @staticmethod
    def normalize_intent_name(intent_name: str) -> str:
        prefix = "Dialogflow_Intent_"
        if intent_name.startswith(prefix):
            return intent_name[len(prefix):].lower()
        return "fallback"

    def detect_intent(self, text: str) -> dict:
        """
        Simulates Dialogflow intent detection API response.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {
                "user_input": text,
                "intent_name": "Default Fallback Intent",
                "confidence": 0.0,
                "response": "I didn't catch that. Can you repeat?"
            }

        # Fast exact pattern match
        if cleaned in self.patterns_dict:
            tag = self.patterns_dict[cleaned]
            return {
                "user_input": text,
                "intent_name": f"Dialogflow_Intent_{tag.capitalize()}",
                "confidence": 1.0,
                "response": random.choice(self.responses_dict[tag])
            }

        user_words = set(cleaned.split())
        best_tag = "Default Fallback Intent"
        best_score = 0.0

        for pattern, tag in self.training_examples:
            p_words = set(pattern.split())

            # Calculate Jaccard similarity & keyword overlap
            intersection = user_words.intersection(p_words)
            union = user_words.union(p_words)
            jaccard = len(intersection) / len(union) if union else 0.0

            # Dialogflow-style local heuristic: reward a phrase that appears
            # verbatim inside the query (or vice versa). More specific
            # multi-word patterns in the curated dataset disambiguate broad
            # one-word phrases such as "course" and "facility".
            if pattern in cleaned or cleaned in pattern:
                jaccard += 0.4

            if jaccard > best_score:
                best_score = jaccard
                best_tag = tag

        # Normalize confidence to [0.0, 1.0]
        confidence = min(round(best_score, 4), 1.0)

        if confidence < 0.05 or best_tag not in self.responses_dict:
            return {
                "user_input": text,
                "intent_name": "Default Fallback Intent",
                "confidence": confidence,
                "response": "I am sorry, as a Dialogflow bot, I could not recognize that university query. Please ask about fees, courses, or hostel facilities."
            }

        response_text = random.choice(self.responses_dict[best_tag])
        return {
            "user_input": text,
            "intent_name": f"Dialogflow_Intent_{best_tag.capitalize()}",
            "confidence": confidence,
            "response": response_text
        }
