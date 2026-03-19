from transformers import pipeline

class IntentClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli", device=-1):
        # device=-1 for CPU, 0 for first GPU
        print(f"Loading Intent Classifier '{model_name}'")
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    def classify_intent(self, text, candidate_labels):
        """Classifies the intent behind the text using zero-shot classification."""
        if not text.strip():
            return None
            
        result = self.classifier(text, candidate_labels)
        return {
            "intent": result["labels"][0],
            "confidence": result["scores"][0],
            "all_scores": dict(zip(result["labels"], result["scores"]))
        }
