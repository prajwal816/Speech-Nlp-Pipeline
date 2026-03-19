import os
import random
import shutil

class AudioDatasetManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def train_test_split(self, test_size=0.2, seed=42):
        """Splits audio files in a directory into train and test sets."""
        random.seed(seed)
        
        all_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    all_files.append(os.path.join(root, file))
        
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - test_size))
        
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]
        
        return train_files, test_files
        
    def organize_split(self, output_dir, train_files, test_files):
        """Copies train/test files into structured directories."""
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        for f in train_files:
            shutil.copy(f, train_dir)
            
        for f in test_files:
            shutil.copy(f, test_dir)
            
        print(f"Organized {len(train_files)} train files and {len(test_files)} test files.")
