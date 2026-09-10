import os
import joblib
import matplotlib.pyplot as plt


def save_model(model, path):
    """
    Save trained model
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load trained model
    """
    return joblib.load(path)


def save_plot(fig, path):
    """
    Save matplotlib figure
    """
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ensure_dir(path):
    """
    Create directory if not exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


def list_audio_files(dataset_path):
    """
    Recursively list all .wav files in dataset
    """
    audio_files = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    return audio_files