import shap
import numpy as np

class NLPExplainer:
    def __init__(self, classifier_pipeline, candidate_labels):
        """
        Initializes SHAP explainer for a zero-shot classification pipeline.
        classifier_pipeline: The underlying HuggingFace pipeline from IntentClassifier.
        """
        self.pipeline = classifier_pipeline
        self.candidate_labels = candidate_labels
        
        # We wrap the pipeline to output merely the scores for the candidate labels 
        # so SHAP can perturb the input text and see how scores change.
        def score_wrapper(texts):
            # pipeline might return a list of dicts for list of texts
            results = self.pipeline(texts, candidate_labels=self.candidate_labels)
            
            # Ensure results is a list even if a single text was provided
            if isinstance(results, dict):
                results = [results]
                
            scores = []
            for res in results:
                # res is a dict with 'labels' and 'scores'
                label2score = dict(zip(res['labels'], res['scores']))
                scores.append([label2score[label] for label in self.candidate_labels])
            return np.array(scores)
            
        self.explainer = shap.Explainer(score_wrapper, self.pipeline.tokenizer)

    def explain(self, text):
        """Generates SHAP values for the given text."""
        shap_values = self.explainer([text])
        return shap_values
