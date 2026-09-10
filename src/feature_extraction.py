import librosa
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_dataset, encode_labels
from utils.config import FEATURE_FILE


def extract_features(file_path):

    try:
        audio, sr = librosa.load(file_path, duration=3)

        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        rms = np.mean(librosa.feature.rms(y=audio))

        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

        features = np.hstack([mfcc, zcr, rms, centroid, bandwidth])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def build_feature_dataset():

    df = preprocess_dataset()

    df, encoder = encode_labels(df)

    features_list = []

    for index, row in df.iterrows():

        features = extract_features(row["file_path"])

        if features is not None:

            features_list.append(
                np.append(features, row["anxiety_label"])
            )

    columns = [f"mfcc_{i}" for i in range(13)] + [
        "zcr",
        "rms",
        "spectral_centroid",
        "spectral_bandwidth",
        "label"
    ]

    feature_df = pd.DataFrame(features_list, columns=columns)

    feature_df.to_csv(FEATURE_FILE, index=False)

    print("\nFeature dataset saved to:", FEATURE_FILE)

    print("\nDataset shape:", feature_df.shape)


if __name__ == "__main__":

    build_feature_dataset()