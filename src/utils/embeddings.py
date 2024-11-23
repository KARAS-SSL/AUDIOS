import os
from typing import List

import pandas as pd
import torch
from multipledispatch import dispatch
from sklearn.utils import resample
from tqdm import tqdm
import librosa

#----------------------------------------------------------------
# EMBEDDING HELPER FUNCTIONS

# Load an audio file and resample it to the target sample rate if necessary.
def load_audio(audio_path: str, target_sample_rate: int):
    waveform, sample_rate = librosa.load(audio_path, sr=None)
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
    dataset_folder_path = "datasets/release"
    dataset_df = pd.read_csv(dataset_meta_path, keep_default_na=False)

    # Load model and processor
    processor = processor_class.from_pretrained(model_id)
    model = model_class.from_pretrained(model_id)
    model.eval()

    print(f"Generating embeddings for {dataset_meta_path} using {model_class.__name__} and {processor_class.__name__}...")

    for filepath in tqdm(dataset_df['file']):
        audio_path = os.path.join(dataset_folder_path, filepath)
        output_path = os.path.join(embeddings_folder_path, f"{os.path.splitext(filepath)[0]}.pt")

        # Load and process audio, then save embeddings
        waveform = load_audio(audio_path, target_sample_rate)
        process_and_save_embeddings(processor, model, waveform, target_sample_rate, output_path)

    print(f"Embeddings saved to {embeddings_folder_path}")


#----------------------------------------------------------------
# EMBEDDING FUNCTIONS

@dispatch(str, int, str, str)
def generate_embeddings_wav2vec(dataset_meta_path: str, target_sample_rate: int, model_id: str, embeddings_folder_path: str) -> None:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    generate_embeddings(dataset_meta_path, target_sample_rate, model_id, embeddings_folder_path, Wav2Vec2ForCTC, Wav2Vec2Processor)

@dispatch(str, int, str, str)
def generate_embeddings_wav2vec2_bert(dataset_meta_path: str, target_sample_rate: int, model_id: str, embeddings_folder_path: str) -> None:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
    generate_embeddings(dataset_meta_path, target_sample_rate, model_id, embeddings_folder_path, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor)
    
@dispatch(str, int, str, str)
def generate_embeddings_hubert(dataset_meta_path: str, target_sample_rate: int, model_id: str, embeddings_folder_path: str) -> None:
    from transformers import Wav2Vec2Processor, HubertForCTC
    generate_embeddings(dataset_meta_path, target_sample_rate, model_id, embeddings_folder_path, HubertForCTC, Wav2Vec2Processor)
