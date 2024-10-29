
import os
import librosa
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.utils import resample

    
#----------------------------------------------------------------------------------------

def main():
    dataset_path = os.path.join(PATH_TO_DATASET, "meta_with_duration.csv")
    dataset_df = pd.read_csv(dataset_path)

    # Load Model 
    processor   = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model       = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    # processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h", padding=True)
    # model     = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    model.eval()  # Set the model to evaluation mode

    output_folder = "../embeddings/xlsr/in_the_wild"
    os.makedirs(output_folder, exist_ok=True)

    for i in tqdm(range(len(dataset_df))):
        audio_name = dataset_df['file'][i]
        audio_path = os.path.join(PATH_TO_DATASET, audio_name)
        output_path = os.path.join(output_folder, f"{audio_name}_embedding.pt")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed (Wav2Vec2 expects 16 kHz)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Convert waveform to model input
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(input_values).logits  # Access the logits or first element

        # Save embeddings
        torch.save(embeddings, output_path)
        print(f"Embeddings saved to {output_path}")

main()
# exploration()
# balance_dataset()
