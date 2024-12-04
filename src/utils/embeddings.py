import os

import librosa
import numpy as np
import pandas as pd
import torch
from multipledispatch import dispatch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import load_audio_file

# ----------------------------------------------------------------
# EMBEDDING HELPER FUNCTIONS

def load_and_resample_audio(filename: str, dataset_folder_path: str, target_sample_rate: int) -> np.ndarray | None:
    """
    Load an audio file and resample it to the target sample rate if necessary.

    Parameters
    ----------
    filename : str
        The name of the audio file.
    dataset_folder_path : str
        The path to the dataset folder.
    target_sample_rate : int
        The target sample rate to resample the audio to.

    Returns
    -------
    waveform : np.ndarray
        The loaded and resampled audio waveform.
    """
    audio = load_audio_file(filename, dataset_folder_path)
    if audio is None:
        return

    waveform, sample_rate = audio
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
    return waveform


def process_and_save_embeddings(processor, model, waveform: np.ndarray, sample_rate: int, output_path: str) -> None:
    """
    Convert audio to embeddings using the specified processor and model, and save the output in a file.

    Parameters
    ----------
    processor : PreTrainedProcessor
        The processor to use for processing the audio.
    model : PreTrainedModel
        The model to use for processing the audio.
    waveform : np.ndarray
        The audio waveform to process.
    sample_rate : int
        The sample rate of the audio waveform.
    output_path : str
        The path to save the embeddings to.

    Returns
    -------
    None
    """
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input_values)
    embeddings = outputs.last_hidden_state

    output_folder_path = os.path.dirname(output_path)
    os.makedirs(output_folder_path, exist_ok=True)
    torch.save(embeddings, output_path)


def validate(metadata_paths: list[str], out_folders: list[str]) -> None:
    """
    Validates that the number of metadata paths matches the number of output folders.

    Parameters
    ----------
    metadata_paths : list[str]
        List of metadata paths.
    out_folders : list[str]
        List of output folders.

    Raises
    ------
    ValueError
        If the number of metadata paths does not match the number of output folders.
    """
    if len(metadata_paths) != len(out_folders):
        raise ValueError("The number of metadata paths must match the number of output folders.")


# ----------------------------------------------------------------
# EMBEDDING GENERATION MAIN FUNCTIONS

def generate_embeddings(
    dataset_folder_path: str,
    dataset_meta_path: str,
    target_sample_rate: int,
    model_id: str,
    embeddings_folder_path: str,
    model_class,
    processor_class,
) -> None:
    """
    Generate the embeddings from a dataset using any model and processor.

    Parameters
    ----------
    dataset_folder_path : str
        The path to the dataset folder.
    dataset_meta_path : str
        The path to the dataset metadata CSV file.
    target_sample_rate : int
        The target sample rate to resample the audio to.
    model_id : str
        The ID of the model to use.
    embeddings_folder_path : str
        The path to save the embeddings to.
    model_class : PreTrainedModel
        The model to use for processing the audio.
    processor_class : PreTrainedProcessor
        The processor to use for processing the audio.

    Returns
    -------
    None
    """
    dataset_df = pd.read_csv(dataset_meta_path, keep_default_na=False)

    # Load model and processor
    processor = processor_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id)

    print(f"Generating embeddings for {dataset_meta_path} using {model_class.__name__} and {processor_class.__name__}...")

    for filepath in tqdm(dataset_df['file']):
        output_path = os.path.join(embeddings_folder_path, f"{os.path.splitext(filepath)[0]}.pt")

        # Load and process audio, then save embeddings
        waveform = load_and_resample_audio(filepath, dataset_folder_path, target_sample_rate)
        if waveform is not None:
            process_and_save_embeddings(processor, model, waveform, target_sample_rate, output_path)

    print(f"Embeddings saved to {embeddings_folder_path}")


def generate_embeddings_multiple(
    dataset_path: str,
    metadata_paths: list[str],
    sr: int,
    model_id: str,
    out_folders: list[str],
    model_class,
    processor_class,
) -> None:
    """
    Handle the generation of embeddings for multiple metadata paths and output folders.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset folder.
    metadata_paths : list[str]
        List of dataset metadata CSV file paths.
    sr : int
        The target sample rate to resample the audio to.
    model_id : str
        The ID of the model to use.
    out_folders : list[str]
        List of output folders for the embeddings.
    model_class : PreTrainedModel
        The model to use for processing the audio.
    processor_class : PreTrainedProcessor
        The processor to use for processing the audio.

    Returns
    -------
    None
    """
    validate(metadata_paths, out_folders)
    for metadata_path, out_folder in zip(metadata_paths, out_folders):
        generate_embeddings(dataset_path, metadata_path, sr, model_id, out_folder, model_class, processor_class)
    print("All embeddings generated!")


# ----------------------------------------------------------------
# MAIN FUNCTIONS TO GENERATE EMBEDDINGS

@dispatch(str, str, int, str, str)
def generate_embeddings_wav2vec(dataset_path: str, metadata_path: str, sr: int, model_id: str, out_folder: str) -> None:
    from transformers import AutoProcessor, Wav2Vec2Model
    generate_embeddings(dataset_path, metadata_path, sr, model_id, out_folder, Wav2Vec2Model, AutoProcessor) 

@dispatch(str, list, int, str, list)
def generate_embeddings_wav2vec(dataset_path: str, metadata_paths: list[str], sr: int, model_id: str, out_folders: list[str]) -> None:
    from transformers import AutoProcessor, Wav2Vec2Model
    generate_embeddings_multiple(dataset_path, metadata_paths, sr, model_id, out_folders, Wav2Vec2Model, AutoProcessor)     

# --------------------------------

@dispatch(str, str, int, str, str)
def generate_embeddings_hubert(dataset_path: str, metadata_path: str, sr: int, model_id: str, out_folder: str) -> None:
    from transformers import AutoProcessor, HubertModel
    generate_embeddings(dataset_path, metadata_path, sr, model_id, out_folder, HubertModel, AutoProcessor)

@dispatch(str, list, int, str, list)
def generate_embeddings_hubert(dataset_path: str, metadata_paths: list[str], sr: int, model_id: str, out_folders: list[str]) -> None:
    from transformers import AutoProcessor, HubertModel
    generate_embeddings_multiple(dataset_path, metadata_paths, sr, model_id, out_folders, HubertModel, AutoProcessor)


# ----------------------------------------------------------------
# EMBEDDINGS LOADER

# Custom dataset for loading voice embeddings and their labels.
class VoiceEmbeddingsDataset(Dataset):
    """
    A custom dataset for loading voice embeddings and their labels.

    Args
    ----
    embeddings_folder_path : str
        Path to the folder containing embeddings.
    gender : str
        Filter embeddings by gender. Default is "" (no filter).
    """

    def __init__(self, embeddings_folder_path: str, gender: str):
        self.data = []

        # Load fake voices (label 0)
        fake_path = os.path.join(embeddings_folder_path, "fake_voices")
        self._load_data(fake_path, label=0, gender=gender)

        # Load real voices (label 1)
        real_path = os.path.join(embeddings_folder_path, "real_voices")
        self._load_data(real_path, label=1, gender=gender)

    def _load_data(self, folder_path: str, label: int, gender: str):
        """
        Load embeddings from the given folder path.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing embeddings.
        label : int
            The label to assign to the embeddings.
        gender : str
            Filter embeddings by gender.
        """
        for person_folder in os.listdir(folder_path):
            person_path = os.path.join(folder_path, person_folder)

            if not os.path.isdir(person_path) or (gender != "" and person_folder.split("_")[1][0] != gender):
                continue  # Skip non-directory files or non-matching gender

            for file in os.listdir(person_path):
                if file.endswith(".pt"):  # Process only .pt files
                    embedding_path = os.path.join(person_path, file)
                    self.data.append((embedding_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        embedding_path, label = self.data[idx]
        embedding = torch.load(embedding_path, weights_only=False)
        embedding_mean = embedding.mean(dim=1).squeeze()  # Squeeze to remove batch dimension

        return embedding_mean, label


def load_embeddings(
    embeddings_folder_path: str,
    gender: str = "",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
) -> list[list[torch.Tensor]]:
    """
    Create a DataLoader for the voice embeddings dataset.

    Parameters
    ----------
    embeddings_folder_path : str
        Path to the folder containing embeddings.
    gender : str
        Filter embeddings by gender. Default is "" (no filter).
    batch_size : int
        Number of samples per batch. Default is 32.
    shuffle : bool
        Whether to shuffle the data. Default is True.
    num_workers : int
        Number of subprocesses to use for data loading. Default is 8.

    Returns
    -------
    list[list[torch.Tensor]]
        List of lists containing embeddings and labels.
    """
    print(f"Loading dataset from {embeddings_folder_path}...")
    dataset = VoiceEmbeddingsDataset(embeddings_folder_path, gender)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return list(tqdm(dataloader, desc="Loading embeddings"))
