import yaml
import os
import sys

# Add project root to path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.audio.processor import AudioProcessor
from src.transcription.asr import WhisperTranscription
from src.nlp.classifier import IntentClassifier
from src.nlp.extractor import EntityExtractor
from src.explainability.explainer import NLPExplainer
from src.explainability.plots import save_shap_plot
from src.pipeline.tracking import ExperimentTracker

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class PipelineRunner:
    def __init__(self, config_path="configs/default.yaml"):
        self.config = load_config(config_path)
        self.tracker = ExperimentTracker(
            experiment_name=self.config['tracking']['experiment_name']
        )
        
        # Initialize modules
        audio_cfg = self.config['audio']
        self.audio_proc = AudioProcessor(sample_rate=audio_cfg['sample_rate'])
        
        trans_cfg = self.config['transcription']
        self.asr = WhisperTranscription(
            model_name=trans_cfg['model_name'], 
            device=trans_cfg['device']
        )
        
        nlp_cfg = self.config['nlp']
        self.intent_classifier = IntentClassifier(model_name=nlp_cfg['intent_model'])
        self.entity_extractor = EntityExtractor(model_name=nlp_cfg['ner_model'])
        
        self.explainer = NLPExplainer(self.intent_classifier.classifier, nlp_cfg['candidate_labels'])

    def run_single(self, audio_path):
        """Runs the pipeline on a single audio file and logs to MLflow."""
        print(f"Running pipeline for {audio_path}")
        
        with self.tracker.start_run(run_name=os.path.basename(audio_path)):
            self.tracker.log_params(self.config['audio'])
            self.tracker.log_params(self.config['transcription'])
            
            # 1. Audio Processing
            audio = self.audio_proc.load_audio(audio_path)
            if self.config['audio']['noise_augmentation']:
                audio = self.audio_proc.add_noise(audio, self.config['audio']['noise_level'])
            
            # 2. Transcription
            print("Transcribing audio...")
            transcript = self.asr.transcribe_audio(audio, language=self.config['transcription']['language'])
            print(f"Transcript: {transcript}")
            
            if not transcript:
                print("Transcription empty, dropping out.")
                return None
                
            # 3. NLP
            intent_res = self.intent_classifier.classify_intent(transcript, self.config['nlp']['candidate_labels'])
            entities = self.entity_extractor.extract_entities(transcript)
            
            print(f"Predicted Intent: {intent_res['intent']} (confidence: {intent_res['confidence']:.2f})")
            print(f"Entities: {entities}")
            
            # Log pseudo-metric for confidence
            self.tracker.log_metrics({"intent_confidence": float(intent_res['confidence'])})
            
            # 4. Explainability
            if self.config['explainability']['enable_shap']:
                print("Generating SHAP explanations...")
                shap_vals = self.explainer.explain(transcript)
                plot_path = f"experiments/shap_{os.path.basename(audio_path)}.png"
                save_shap_plot(shap_vals, self.config['nlp']['candidate_labels'], plot_path)
                self.tracker.log_artifact(plot_path)

            return {
                "transcript": transcript,
                "intent": intent_res,
                "entities": entities
            }

if __name__ == "__main__":
    runner = PipelineRunner()
    # Provide a real path to test
    # runner.run_single("data/test/sample.wav")
