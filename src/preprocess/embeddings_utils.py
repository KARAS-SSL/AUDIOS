
import os
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.utils import resample

def generate_embeddings_wav2vec(dataset_path, target_sample_rate, model_id, embeddings_path):
    dataset_df = pd.read_csv(dataset_path)
   
    # Load Model in evaluation mode
    processor   = Wav2Vec2Processor.from_pretrained(model_id)
    model       = Wav2Vec2ForCTC.from_pretrained(model_id) 
    model.eval()

    for i in tqdm(range(len(dataset_df))):
        audio_name  = dataset_df['file'][i]
        audio_path  = os.path.join(os.path.dirname(dataset_path), audio_name)
        output_path = os.path.join(embeddings_path, f"{audio_name}_embedding.pt")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed (Wav2Vec2 expects 16 kHz)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

        # Convert waveform to model input
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_values

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(input_values).logits  # Access the logits or first element

        # Save embeddings
        torch.save(embeddings, output_path)
        print(f"Embeddings saved to {output_path}") 
    
