import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """Loads audio file and resamples if necessary."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio

    def segment_audio(self, audio, segment_length_sec):
        """Segments audio into fixed-length chunks."""
        segment_length_samples = int(self.sample_rate * segment_length_sec)
        segments = []
        for i in range(0, len(audio), segment_length_samples):
            segment = audio[i:i + segment_length_samples]
            if len(segment) < segment_length_samples:
                # Pad with zeros if shorter
                segment = np.pad(segment, (0, segment_length_samples - len(segment)))
            segments.append(segment)
        return segments

    def remove_silence(self, audio, top_db=20):
        """Removes silence from audio."""
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(non_silent_intervals) > 0:
            non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
            return non_silent_audio
        return audio

    def add_noise(self, audio, noise_level=0.005):
        """Injects white noise into audio."""
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_level * noise
        return augmented_audio
