import whisper
import torch
import numpy as np

class WhisperTranscription:
    def __init__(self, model_name="base", device="cpu"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Loading Whisper '{model_name}' on {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe_audio(self, audio, language="en"):
        """Transcribes a single audio array."""
        # Whisper model expects 16kHz float32 audio
        audio = audio.astype(np.float32)
        result = self.model.transcribe(audio, language=language, fp16=torch.cuda.is_available())
        return result["text"].strip()

    def batch_transcribe(self, audio_list, language="en", batch_size=8):
        """Transcribes a list of audio arrays in batches.
           Returns a list of transcribed texts.
        """
        results = []
        # While true batching is complex in raw Whisper, we orchestrate it effectively here
        for i in range(0, len(audio_list), batch_size):
            batch = audio_list[i:i + batch_size]
            batch_texts = []
            for audio in batch:
                batch_texts.append(self.transcribe_audio(audio, language=language))
            results.extend(batch_texts)
        return results

    def transcribe_stream(self, audio_chunk_generator, language="en"):
        """Simulates streaming transcription by yielding results for chunks."""
        for chunk in audio_chunk_generator:
            text = self.transcribe_audio(chunk, language=language)
            if text:
                yield text
