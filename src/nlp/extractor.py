from transformers import pipeline

class EntityExtractor:
    def __init__(self, model_name="dslim/bert-base-NER", device=-1):
        print(f"Loading Entity Extractor '{model_name}'")
        self.ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)

    def extract_entities(self, text):
        """Extracts named entities from the text."""
        if not text.strip():
            return []
            
        entities = self.ner_pipeline(text)
        return [
            {
                "entity_group": ent["entity_group"],
                "word": ent["word"],
                "score": float(ent["score"])
            }
            for ent in entities
        ]
