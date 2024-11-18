
import csv
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm

tqdm.pandas()

#----------------------------------------------------------------
# Function to get an audio duration
def audio_duration(filename, dataset_path):
    try:
        audio_path = os.path.join(dataset_path, filename)
        y, sr = librosa.load(audio_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"Could not load file {filename}: {e}")
        return None

# Function to get an audio amplitude
def audio_amplitude(filename, dataset_path):
    try:
        audio_path = os.path.join(dataset_path, filename)
        y, sr = librosa.load(audio_path, sr=None)
        return np.max(np.abs(y))
    except Exception as e:
        print(f"Could not load file {filename}: {e}")
        return None

#----------------------------------------------------------------
# Function to generate a csv file from a dataset
# Create a list containing the path, speaker's name, id and gender, and label for each audio file.
def generate_dataset_file(dataset_path):
    dataset_folder_path = os.path.dirname(dataset_path)
    fake_audios_path    = os.path.join(dataset_folder_path, "fake_voices")
    real_audios_path    = os.path.join(dataset_folder_path, "real_voices")

    files = []

    print("Generating dataset file...")

    # For every spoofed file, add its metadata to the list
    print("spoof files: ", end="")
    for folder in tqdm(os.listdir(fake_audios_path)):
        folder_path = os.path.join(fake_audios_path, folder)

        person, ids, *_ = folder.split("_")
        gender = ids[0]
        for filename in os.listdir(folder_path):
            audio_path = os.path.join("fake_voices", folder, filename)  # relative path inside the dataset
            files.append([audio_path, person, ids, gender, "spoof"])

    print("bona-fide files: ", end="")
    # For every bona-fide file, add its metadata to the list
    for folder in tqdm(os.listdir(real_audios_path)):
        folder_path = os.path.join(real_audios_path, folder)

        person, ids, *_ = folder.split("_")
        gender = ids[0]
        for filename in os.listdir(folder_path):
            audio_path = os.path.join("real_voices", folder, filename)
            files.append([audio_path, person, ids, gender, "bona-fide"])

    # Export the list to a .csv file.
    print("Writing to disk...", end=" ")    
    fields = ["file", "speaker", "id", "gender", "label"]
    with open(dataset_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(files)
    print("Done!")

    print("Dataset file generated. Dataset saved to ", dataset_path)

# Function to add duration information to the dataset
def add_duration_dataset(dataset_path, new_dataset_path):
    dataset_folder_path    = os.path.dirname(dataset_path)
    dataset_df             = pd.read_csv(dataset_path)
    dataset_df['duration'] = dataset_df['file'].progress_apply(lambda filename: audio_duration(filename, dataset_folder_path))
    dataset_df.to_csv(new_dataset_path, index=False)
    print("Duration added to dataset. New dataset saved to ", new_dataset_path)

# Function to add amplitude information to the dataset
def add_amplitude_dataset(dataset_path, new_dataset_path):
    dataset_folder_path     = os.path.dirname(dataset_path)
    dataset_df              = pd.read_csv(dataset_path)
    dataset_df['amplitude'] = dataset_df['file'].progress_apply(lambda filename: audio_amplitude(filename, dataset_folder_path))
    dataset_df.to_csv(new_dataset_path, index=False)
    print("Amplitude added to dataset. New dataset saved to ", new_dataset_path)

#----------------------------------------------------------------
# Function to normalize audio to a target RMS energy level
def rms_normalize(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio  # Avoid division by zero
    return audio * (target_rms / rms)

# Function to normalize audio to have a peak amplitude of 1.
def peak_normalize(audio):
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio  # Avoid division by zero
    return audio / peak

def normalize_audio_file(filename, dataset_path, output_path):
    try:
        audio_path = os.path.join(dataset_path, filename)
        y, sr = librosa.load(audio_path, sr=None)

        normalized_audio = peak_normalize(y)

        output_path = os.path.join(output_path, os.path.dirname(filename))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file = os.path.join(output_path, os.path.basename(filename))
        sf.write(output_file, normalized_audio, sr)
        # print(f"Normalized audio saved to {output_file}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Function to normalize the audio amplitude of the dataset
def normalize_dataset(dataset_path, new_dataset_path):
    dataset_folder_path     = os.path.dirname(dataset_path)
    new_dataset_folder_path = os.path.dirname(new_dataset_path)
    dataset_df              = pd.read_csv(dataset_path)

    print("Normalizing dataset...")
    # Normalize all audio files
    for filename in tqdm(dataset_df['file']):
        normalize_audio_file(filename, dataset_folder_path, new_dataset_folder_path)
    print("Done!")

    # Generate a new csv file
    generate_dataset_file(new_dataset_path)

    print("Dataset normalized. New dataset saved to ", new_dataset_path)

#----------------------------------------------------------------
# Balances the dataset by removing samples with low duration
def balance_dataset(dataset_path, new_dataset_path, imbalance_threshold, seed):
    dataset_df = pd.read_csv(dataset_path)
    
    # Per speaker counts
    spoof_per_speaker = dataset_df[dataset_df.label == 'spoof'].groupby("speaker").duration.count()
    bonafide_per_speaker = dataset_df[dataset_df.label == 'bona-fide'].groupby("speaker").duration.count()

    # Combine counts into a single DataFrame and fill NaN with 0
    counts_df = pd.DataFrame({
        'spoof_count': spoof_per_speaker,
        'bona_fide_count': bonafide_per_speaker
    }).fillna(0)
    
    # Calculate total count and spoof ratio for each speaker
    counts_df['total'] = counts_df['spoof_count'] + counts_df['bona_fide_count']
    counts_df['spoof_ratio'] = counts_df['spoof_count'] / counts_df['total']

    # Initialize list to hold balanced data
    final_data = []
    
    for speaker in counts_df.index:
        speaker_data = dataset_df[dataset_df['speaker'] == speaker]
        
        # Separate spoof and bonafide samples
        spoof_samples = speaker_data[speaker_data['label'] == 'spoof']
        bonafide_samples = speaker_data[speaker_data['label'] == 'bona-fide']
        
        # Calculate the spoof ratio for this speaker
        spoof_ratio = counts_df.loc[speaker, 'spoof_ratio']
        
        # Check if the speaker's spoof/bona-fide ratio is within the threshold
        if spoof_ratio < imbalance_threshold or spoof_ratio > (1 - imbalance_threshold):
            # Balance spoof and bona-fide samples by truncating the excess
            target_count = min(len(spoof_samples), len(bonafide_samples))
            spoof_samples = spoof_samples.sample(n=target_count, random_state=seed)
            bonafide_samples = bonafide_samples.sample(n=target_count, random_state=seed)

        # Append both spoof and bona-fide samples for this speaker
        final_data.append(pd.concat([spoof_samples, bonafide_samples]))

    # Concatenate all balanced samples and sort by numeric part of 'file' column
    balanced_dataset_df = pd.concat(final_data)
    balanced_dataset_df['file_index'] = balanced_dataset_df['file'].str.extract(r'(\d+)').astype(int)
    balanced_dataset_df = balanced_dataset_df.sort_values(by='file_index').drop(columns='file_index').reset_index(drop=True)
    
    balanced_dataset_df.to_csv(new_dataset_path, index=False)
    print("Balanced dataset saved to", new_dataset_path)

#----------------------------------------------------------------
    
def display_info_dataset(dataset_path):
    print("Dataset info ------") 
    # Read dataset
    dataset_df             = pd.read_csv(dataset_path)
    if 'duration' not in dataset_df:
        print("[ERROR] Duration column not found. Please add it to the dataset.")
        return 
    
    # Dataset duration
    print(dataset_df.groupby("label")['duration'].sum().sort_values(ascending=False) / 3600, end="\n\n")

    # Dataset all info
    print(dataset_df) 

    # Spoof and bonafide per speaker
    spoof_per_speaker    = dataset_df[dataset_df.label == 'spoof'].groupby("speaker").duration.count()
    bonafide_per_speaker = dataset_df[dataset_df.label == 'bona-fide'].groupby("speaker").duration.count()
    counts_df = pd.DataFrame({
        'spoof_count': spoof_per_speaker,
        'bona_fide_count': bonafide_per_speaker
    }).fillna(0)  
    counts_df['total']       = counts_df['spoof_count'] + counts_df['bona_fide_count']
    counts_df['spoof_ratio'] = counts_df['spoof_count'] / counts_df['total']
    counts_df.sort_values("spoof_ratio")
    print(counts_df) 
