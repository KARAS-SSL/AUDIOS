import os
from typing import List

import pandas as pd
import torch
from multipledispatch import dispatch
from sklearn.utils import resample
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa


# Function to generate embeddings for a given dataset
@dispatch(str, int, str, str)
def generate_embeddings_wav2vec(dataset_meta_path: str, target_sample_rate: int, model_id: str, embeddings_folder_path: str) -> None:
    dataset_folder_path = os.path.dirname(dataset_meta_path)
    dataset_df          = pd.read_csv(dataset_meta_path, keep_default_na=False)

    # Load Model in evaluation mode
    processor  = Wav2Vec2Processor.from_pretrained(model_id)
    model      = Wav2Vec2ForCTC.from_pretrained(model_id) 
    model.eval()

    print(f"Generating embeddings for {dataset_meta_path}...")

    for filepath in tqdm(dataset_df['file']):
        audio_path  = os.path.join(dataset_folder_path, filepath)
        output_path = os.path.join(embeddings_folder_path, f"{os.path.splitext(filepath)[0]}.pt")

        # Load audio
        waveform, sample_rate = librosa.load(audio_path, sr=None)  # Load with the original sample rate

        # Resample if needed (Wav2Vec2 expects 16 kHz)
        if sample_rate != target_sample_rate:
          waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)

        # Convert waveform to model input
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_values

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(input_values).logits  # Access the logits or first element

        # Save embeddings
        output_folder_path = os.path.dirname(output_path)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)

        torch.save(embeddings, output_path)
    print(f"Embeddings saved to {embeddings_folder_path}") 


# Function to generate embeddings for a list of datasets
@dispatch(list, int, str, str)
def generate_embeddings_wav2vec(dataset_meta_path: List[str], target_sample_rate: int, model_id: str, embeddings_folder_path: str) -> None:
    for meta_path in dataset_meta_path:
        generate_embeddings_wav2vec(meta_path, target_sample_rate, model_id, embeddings_folder_path)
    
    print("All embeddings generated!")
