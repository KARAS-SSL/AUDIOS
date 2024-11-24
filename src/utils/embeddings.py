import os
from typing import List, Callable

import librosa
import numpy as np
import pandas as pd
import torch
from multipledispatch import dispatch
from sklearn.utils import resample
from tqdm import tqdm

from .dataset import load_audio_file

#----------------------------------------------------------------
# EMBEDDING HELPER FUNCTIONS

# Load an audio file and resample it to the target sample rate if necessary.
def load_audio(filename: str, dataset_folder_path: str, target_sample_rate: int) -> np.ndarray | None:
    audio = load_audio_file(filename, dataset_folder_path)
    if audio is None:
        return
    
    waveform, sample_rate = audio
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
    return waveform

# Convert audio to embeddings using the specified processor and model, and save the output.
def process_and_save_embeddings(processor, model, waveform, sample_rate, output_path):
    input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values

    with torch.no_grad():
        embeddings = model(input_values).logits  # Access logits for embeddings

    output_folder_path = os.path.dirname(output_path)
    os.makedirs(output_folder_path, exist_ok=True)
    torch.save(embeddings, output_path)

#----------------------------------------------------------------
# EMBEDDING MAIN FUNCTION

# Generalized function to generate embeddings using any model and processor.

def generate_embeddings(
    dataset_folder_path: str,
    dataset_meta_path: str, 
    target_sample_rate: int, 
    model_id: str, 
    embeddings_folder_path: str, 
    model_class, 
    processor_class
) -> None:
    """
    Generalized function to generate embeddings using any model and processor.
    """
    dataset_df = pd.read_csv(dataset_meta_path, keep_default_na=False)

    # Load model and processor
    processor = processor_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id)
    model.eval()

    print(f"Generating embeddings for {dataset_meta_path} using {model_class.__name__} and {processor_class.__name__}...")

    for filepath in tqdm(dataset_df['file']):
        output_path = os.path.join(embeddings_folder_path, f"{os.path.splitext(filepath)[0]}.pt")

        # Load and process audio, then save embeddings
        waveform = load_audio(filepath, dataset_folder_path, target_sample_rate)
        if waveform is not None:
            process_and_save_embeddings(processor, model, waveform, target_sample_rate, output_path)

    print(f"Embeddings saved to {embeddings_folder_path}")


#----------------------------------------------------------------
# EMBEDDING FUNCTIONS

# Auxiliary function to validate that the number of metadata paths matches the number of output folders.

def validate(metadata_paths: List[str], out_folders: List[str]) -> None:
    """Validates that the number of metadata paths matches the number of output folders."""
    if len(metadata_paths) != len(out_folders):
        raise ValueError("The number of metadata paths must match the number of output folders.")

def generate_embeddings_multiple(
    dataset_path: str, metadata_paths: List[str], fr: int, model_id: str, out_folders: List[str], 
    model_class, processor_class
) -> None:
    """
    Handles the generation of embeddings for multiple metadata paths and output folders.
    """
    validate(metadata_paths, out_folders)
    for metadata_path, out_folder in zip(metadata_paths, out_folders):
        generate_embeddings(dataset_path, metadata_path, fr, model_id, out_folder, model_class, processor_class)
    print("All embeddings generated!")

# Main functions to generate embeddings #

@dispatch(str, str, int, str, str)
def generate_embeddings_wav2vec(dataset_path:str, metadata_path:str, fr:int, model_id:str, out_folder:str) -> None:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor    
    generate_embeddings(dataset_path, metadata_path, fr, model_id, out_folder, 
                        Wav2Vec2ForSequenceClassification, Wav2Vec2Processor)

@dispatch(str, list, int, str, list)
def generate_embeddings_wav2vec(dataset_path:str, metadata_paths:List[str], fr:int, model_id:str, out_folders:List[str]):
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor    
    generate_embeddings_multiple(dataset_path, metadata_paths, fr, model_id, out_folders, 
                                 Wav2Vec2ForSequenceClassification, Wav2Vec2Processor)
    
@dispatch(str, str, int, str, str)
def generate_embeddings_wav2vec2_bert(dataset_path:str, metadata_path:str, fr:int, model_id:str, out_folder:str) -> None:
    from transformers import Wav2Vec2BertForSequenceClassification, Wav2Vec2Processor
    generate_embeddings(dataset_path, metadata_path, fr, model_id,out_folder, 
                        Wav2Vec2BertForSequenceClassification, Wav2Vec2Processor)

@dispatch(str, list, int, str, list)
def generate_embeddings_wav2vec2_bert(dataset_path:str, metadata_paths:List[str], fr:int, model_id:str, out_folders:List[str]):
    from transformers import Wav2Vec2BertForSequenceClassification, Wav2Vec2Processor
    generate_embeddings_multiple(dataset_path, metadata_paths, fr, model_id, out_folders, 
                                 Wav2Vec2BertForSequenceClassification, Wav2Vec2Processor)

@dispatch(str, str, int, str, str)
def generate_embeddings_hubert(dataset_path: str, metadata_path: str, fr: int, model_id: str, out_folder: str) -> None:
    from transformers import HubertForSequenceClassification, AutoFeatureExtractor
    generate_embeddings(dataset_path, metadata_path, fr, model_id, out_folder, HubertForSequenceClassification, AutoFeatureExtractor)

@dispatch(str, list, int, str, list)
def generate_embeddings_hubert(dataset_path: str, metadata_paths: List[str], fr: int, model_id: str, out_folders: List[str]):
    from transformers import HubertForSequenceClassification, AutoFeatureExtractor
    generate_embeddings_multiple(dataset_path, metadata_paths, fr, model_id, out_folders, HubertForSequenceClassification, AutoFeatureExtractor)    
