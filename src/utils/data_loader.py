import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import librosa
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class GTZANDataset(Dataset):
    """
    GTZAN Dataset for music clustering
    """
    def __init__(self, features, labels=None, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        
        if self.labels is not None:
            return x, self.labels[idx]
        return x


class HybridDataset(Dataset):
    """
    Hybrid dataset for audio + lyrics
    """
    def __init__(self, audio_features, lyrics_features, labels=None):
        self.audio_features = torch.FloatTensor(audio_features)
        self.lyrics_features = torch.FloatTensor(lyrics_features)
        self.labels = labels
    
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        audio = self.audio_features[idx]
        lyrics = self.lyrics_features[idx]
        
        if self.labels is not None:
            return audio, lyrics, self.labels[idx]
        return audio, lyrics


def load_gtzan_csv(csv_path, genre_filter=None):
    """
    Load GTZAN dataset from CSV file
    
    Args:
        csv_path: Path to CSV file with features
        genre_filter: List of genres to include (None = all)
    
    Returns:
        features, labels, genre_names
    """
    df = pd.read_csv(csv_path)
    
    # Filter genres if specified
    if genre_filter:
        df = df[df['label'].isin(genre_filter)]
    
    # Extract labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'].values) #type: ignore
    genre_names = label_encoder.classes_
    
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ['filename', 'label', 'length']]
    features = df[feature_cols].values
    
    return features, labels, genre_names


def load_gtzan_with_lyrics(csv_path, max_lyrics_len=512):
    """
    Load GTZAN dataset with lyrics
    
    Args:
        csv_path: Path to CSV file with features and lyrics
        max_lyrics_len: Maximum length for lyrics embeddings
    
    Returns:
        audio_features, lyrics_features, labels, genre_names
    """
    df = pd.read_csv(csv_path)
    
    # Extract labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['genre'].values) #type: ignore
    genre_names = label_encoder.classes_
    
    # Extract audio features (numeric columns only)
    audio_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    audio_features = df[audio_cols].values
    
    # Process lyrics if available
    if 'lyrics_clean' in df.columns:
        # Simple bag-of-words representation
        
        vectorizer = TfidfVectorizer(max_features=max_lyrics_len, stop_words='english')
        lyrics_features = vectorizer.fit_transform(df['lyrics_clean'].fillna('')).toarray() #type: ignore
    else:
        # Use dummy features if no lyrics
        lyrics_features = np.zeros((len(df), max_lyrics_len))
    
    return audio_features, lyrics_features, labels, genre_names


def extract_audio_features(audio_path, sr=22050, n_mfcc=20):
    """
    Extract audio features from audio file
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        features: Dictionary of audio features
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Extract features
    features = {}
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    features['chroma_std'] = np.std(chroma, axis=1)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
    # Flatten features
    feature_vector = []
    for key in sorted(features.keys()):
        value = features[key]
        if isinstance(value, np.ndarray):
            feature_vector.extend(value)
        else:
            feature_vector.append(value)
    
    return np.array(feature_vector)


def create_spectrogram(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Create mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    
    return log_mel_spec


def prepare_data(features, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare data for training
    
    Args:
        features: Feature matrix
        labels: Label vector
        test_size: Test set size
        val_size: Validation set size
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = counts.min()
    
    # Use stratified split only if all classes have at least 2 samples
    use_stratify = min_samples >= 2
    
    if not use_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s). Using non-stratified split.")
        print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, 
        stratify=labels if use_stratify else None
    )
    
    # Check if we can stratify the validation split
    unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
    use_stratify_val = counts_temp.min() >= 2
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, 
        stratify=y_temp if use_stratify_val else None
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    train_dataset = GTZANDataset(X_train, y_train)
    val_dataset = GTZANDataset(X_val, y_val)
    test_dataset = GTZANDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
