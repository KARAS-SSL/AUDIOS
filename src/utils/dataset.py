import os
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()


# ----------------------------------------------------------------
# AUDIO FILE PROPERTIES FUNCTIONS

# Function to load an audio file
def load_audio_file(filename: str, dataset_folder_path: str) -> Tuple[np.ndarray, int | float] | None:
    try:
        audio_path = os.path.join(dataset_folder_path, filename)
        return librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Could not load file {filename}: {e}")
        return None


# Function to get an audio duration
def audio_duration(filename: str, dataset_folder_path: str) -> float | None:
    audio = load_audio_file(filename, dataset_folder_path)
    if audio is not None:
        y, sr = audio
        return librosa.get_duration(y=y, sr=sr)


# Function to get an audio amplitude
def audio_amplitude(filename: str, dataset_folder_path: str) -> float | None:
    audio = load_audio_file(filename, dataset_folder_path)
    if audio is not None:
        y, sr = audio
        return np.max(np.abs(y))


# ----------------------------------------------------------------
# DATASET CSV FUNCTIONS

# Function to generate a CSV file from a dataset
# Create a list containing the path, speaker's name, id and gender, and label for each audio file.
def generate_dataset_files_meta(dataset_folder_path: str):
    fake_audios_path = os.path.join(dataset_folder_path, "fake_voices")
    real_audios_path = os.path.join(dataset_folder_path, "real_voices")
    files = []

    # Process fake audios
    for folder in os.listdir(fake_audios_path):
        folder_path = os.path.join(fake_audios_path, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directories

        try:
            speaker_name, speaker_id, *_ = folder.split("_")
            speaker_gender = speaker_id[0]
        except ValueError:
            print(f"Skipping folder with unexpected format: {folder}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".wav"):
                continue  # Skip non-wav files
            audio_path = os.path.join("fake_voices", folder, filename)  # relative path inside the dataset
            files.append([audio_path, speaker_name, speaker_id, speaker_gender, "spoof"])

    # Process real audios
    for folder in os.listdir(real_audios_path):
        folder_path = os.path.join(real_audios_path, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            speaker_name, speaker_id, *_ = folder.split("_")
            speaker_gender = speaker_id[0]
        except ValueError:
            print(f"Skipping folder with unexpected format: {folder}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".wav"):
                continue
            audio_path = os.path.join("real_voices", folder, filename)
            files.append([audio_path, speaker_name, speaker_id, speaker_gender, "bona-fide"])

    # Write metadata to CSV
    fields = ["file", "speaker", "id", "gender", "label"]
    output_file = os.path.join(dataset_folder_path, "files-metadata.csv")
    pd.DataFrame(files, columns=fields).sort_values("file").to_csv(output_file, index=False)
    print(f"Metadata CSV written to {output_file}")


# Function to generate a CSV file from a dataset
# Create a list containing the name, id, gender, number of spoofed and bona-fide files and path to said files for each person.
def generate_dataset_people_meta(dataset_folder_path: str) -> None:
    fake_audios_path = os.path.join(dataset_folder_path, "fake_voices")
    real_audios_path = os.path.join(dataset_folder_path, "real_voices")
    people = {}

    # Process fake voices
    for folder in os.listdir(fake_audios_path):
        path = os.path.join(fake_audios_path, folder)
        if not os.path.isdir(path):
            continue  # Skip non-folder items

        files = [f for f in os.listdir(path) if f.endswith(".wav")]

        try:
            speaker_name, speaker_id, *_ = folder.split("_")
            speaker_gender = speaker_id[0]
        except ValueError:
            print(f"Skipping folder with unexpected format: {folder}")
            continue

        people[speaker_name] = {
            "gender": speaker_gender,
            "id": speaker_id,
            "spoof_count": len(files),
            "spoof_folder": os.path.join("fake_voices", folder),
            "bonafide_count": 0,
            "bonafide_folder": "",
        }

    # Process real voices
    for folder in os.listdir(real_audios_path):
        path = os.path.join(real_audios_path, folder)
        if not os.path.isdir(path):
            continue  # Skip non-folder items

        files = [f for f in os.listdir(path) if f.endswith(".wav")]

        try:
            speaker_name, speaker_id, *_ = folder.split("_")
            speaker_gender = speaker_id[0]
        except ValueError:
            print(f"Skipping folder with unexpected format: {folder}")
            continue

        if speaker_name in people:
            people[speaker_name]["bonafide_count"] = len(files)
            people[speaker_name]["bonafide_folder"] = os.path.join("real_voices", folder)
        else:
            people[speaker_name] = {
                "gender": speaker_gender,
                "id": speaker_id,
                "spoof_count": 0,
                "spoof_folder": "",
                "bonafide_count": len(files),
                "bonafide_folder": os.path.join("real_voices", folder),
            }

    # Convert the dictionary to a list of lists
    people = [[k, v["gender"], v["id"], v["spoof_count"], v["bonafide_count"], v["spoof_folder"], v["bonafide_folder"]] for k, v in people.items()]

    # Write metadata to CSV
    fields = ["person", "gender", "id", "spoof_count", "bonafide_count", "spoof_folder", "bonafide_folder"]
    output_file = os.path.join(dataset_folder_path, "people-metadata.csv")
    pd.DataFrame(people, columns=fields).sort_values("person").to_csv(output_file, index=False)
    print(f"Metadata CSV written to {output_file}")


# Function to calculate the duration of each audio file
def calculate_durations(dataset_meta_path: str) -> pd.DataFrame:
    dataset_folder_path = os.path.dirname(dataset_meta_path)
    dataset_df          = pd.read_csv(dataset_meta_path, keep_default_na=False)

    print("Calculating durations...")
    dataset_df["duration"] = dataset_df["file"].progress_apply(
        lambda filename: audio_duration(filename, dataset_folder_path)
    )
    print("Durations calculation done.")
    return dataset_df


# Function to add duration information to the dataset's metadata
def add_duration_dataset(dataset_meta_path: str, new_dataset_meta_path: str) -> None:
    dataset_df = calculate_durations(dataset_meta_path)
    dataset_df.to_csv(new_dataset_meta_path, index=False)
    print("Duration added to dataset. New dataset saved to ", new_dataset_meta_path)


# Function to calculate the amplitude of each audio file
def calculate_amplitudes(dataset_meta_path: str) -> pd.DataFrame:
    dataset_folder_path = os.path.dirname(dataset_meta_path)
    dataset_df          = pd.read_csv(dataset_meta_path, keep_default_na=False)

    print("Calculating amplitudes...")
    dataset_df["amplitude"] = dataset_df["file"].progress_apply(
        lambda filename: audio_amplitude(filename, dataset_folder_path)
    )
    print("Amplitudes calculation done.")
    return dataset_df


# Function to add amplitude information to the dataset's metadata
def add_amplitude_dataset(dataset_meta_path: str, new_dataset_meta_path: str) -> None:
    dataset_df = calculate_amplitudes(dataset_meta_path)
    dataset_df.to_csv(new_dataset_meta_path, index=False)
    print("Amplitude added to dataset. New dataset saved to ", new_dataset_meta_path)


# ----------------------------------------------------------------
# NORMALIZATION FUNCTIONS

# Function to normalize audio to a target RMS energy level
def rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio  # Avoid division by zero
    return audio * (target_rms / rms)


# Function to normalize audio to have a peak amplitude of 1.
def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio  # Avoid division by zero
    return audio / peak


# Function to normalize an audio file
def normalize_audio_file(filename: str, dataset_folder_path: str, output_path: str, target_rms: float | None = None) -> None:
    audio = load_audio_file(filename, dataset_folder_path)

    if audio is None:
        return

    y, sr = audio

    if target_rms is not None:
        normalized_audio = rms_normalize(y, target_rms)
    else:
        normalized_audio = peak_normalize(y)

    output_path = os.path.join(output_path, os.path.dirname(filename))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, os.path.basename(filename))
    sf.write(output_file, normalized_audio, sr)


# Function to normalize the audio amplitude of the dataset
def normalize_dataset(dataset_meta_path: str, new_dataset_folder_path: str, target_rms: float | None = None) -> None:
    dataset_folder_path = os.path.dirname(dataset_meta_path)
    dataset_df          = pd.read_csv(dataset_meta_path, keep_default_na=False)

    # Normalize all audio files
    print("Normalizing dataset...")
    for filename in tqdm(dataset_df["file"]):
        normalize_audio_file(filename, dataset_folder_path, new_dataset_folder_path, target_rms)
    print("Done!")

    # Generate new csv files for the normalized dataset
    generate_dataset_people_meta(new_dataset_folder_path)
    generate_dataset_files_meta(new_dataset_folder_path)

    print("Dataset normalized. New dataset saved to ", new_dataset_folder_path)


#----------------------------------------------------------------
# DATASET SPLIT FUNCTIONS

# Function to split a dataset in pretext and downstream
def split_dataset(
    full_df: pd.DataFrame,
    pretext_train_percentage: float,
    pretext_val_percentage: float,
    downstream_train_percentage: float,
    downstream_val_percentage: float,
    downstream_test_percentage: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split dataset in pretext and downstream
    pretext_percentage = pretext_train_percentage + pretext_val_percentage
    pretext_df, downstream_df = train_test_split(full_df, train_size=pretext_percentage, random_state=random_state)

    # Split pretext in train and val
    pretext_train_proportion = pretext_train_percentage / pretext_percentage
    pretext_train_df, pretext_val_df = train_test_split(pretext_df, train_size=pretext_train_proportion, random_state=random_state)

    # Split downstream in train, val and test
    downstream_percentage = downstream_train_percentage + downstream_val_percentage + downstream_test_percentage
    downstream_train_proportion = downstream_train_percentage / downstream_percentage
    downstream_train_df, downstream_val_test_df = train_test_split(downstream_df, train_size=downstream_train_proportion, random_state=random_state)

    downstream_val_proportion = downstream_val_percentage / (downstream_val_percentage + downstream_test_percentage)
    downstream_val_df, downstream_test_df = train_test_split(downstream_val_test_df, train_size=downstream_val_proportion, random_state=random_state)

    return pretext_train_df, pretext_val_df, downstream_train_df, downstream_val_df, downstream_test_df


# Function to split the full dataset
def split_full_dataset(
    people_dataset_meta_path: str,
    files_dataset_meta_path: str,
    pre_train_percentage: float,
    pre_val_percentage: float,
    down_train_percentage: float,
    down_val_percentage: float,
    down_test_percentage: float,
    random_state: int
) -> None:
    print(f"Splitting dataset {people_dataset_meta_path}...") 

    if(pre_train_percentage + pre_val_percentage + down_train_percentage + down_val_percentage + down_test_percentage != 1):
        raise Exception("Sum of pretain train and val and downstream train, val and test percentages must be equal to 1")

    # Full dataset
    full_df     = pd.read_csv(people_dataset_meta_path, keep_default_na=False)
    spoofed_df  = full_df.loc[full_df['spoof_count'] > 0]
    no_spoof_df = full_df.loc[full_df['spoof_count'] == 0]  # people with no spoof audios are separated for the downstream test

    # Split full dataset in male and female to make sure training datasets are balanced
    male_df   = spoofed_df.loc[spoofed_df['gender'] == 'M']
    female_df = spoofed_df.loc[spoofed_df['gender'] == 'F']

    # Split male and female datasets in pretext train, pretext val, downstream train, downstream val and downstream test
    male_split   = split_dataset(male_df, pre_train_percentage, pre_val_percentage, down_train_percentage, down_val_percentage, down_test_percentage, random_state)
    female_split = split_dataset(female_df, pre_train_percentage, pre_val_percentage, down_train_percentage, down_val_percentage, down_test_percentage, random_state)

    # Merge male and female split datasets
    people_pretext_train_df    = pd.concat([male_split[0], female_split[0]])
    people_pretext_val_df      = pd.concat([male_split[1], female_split[1]])
    people_downstream_train_df = pd.concat([male_split[2], female_split[2]])
    people_downstream_val_df   = pd.concat([male_split[3], female_split[3]])
    people_downstream_test_df  = pd.concat([male_split[4], female_split[4], no_spoof_df])

    # Split files dataset based on people dataset
    files_full_df             = pd.read_csv(files_dataset_meta_path, keep_default_na=False)
    files_pretext_train_df    = files_full_df.loc[files_full_df['id'].isin(people_pretext_train_df['id'])]
    files_pretext_val_df      = files_full_df.loc[files_full_df['id'].isin(people_pretext_val_df['id'])]
    files_downstream_train_df = files_full_df.loc[files_full_df['id'].isin(people_downstream_train_df['id'])]
    files_downstream_val_df   = files_full_df.loc[files_full_df['id'].isin(people_downstream_val_df['id'])]
    files_downstream_test_df  = files_full_df.loc[files_full_df['id'].isin(people_downstream_test_df['id'])]

    # Save pretext and downstream datasets
    dataset_folder               = os.path.join(os.path.dirname(people_dataset_meta_path), "splits")
    dataset_folder_by_people     = os.path.join(dataset_folder, "by_people")
    dataset_folder_by_file       = os.path.join(dataset_folder, "by_file")
    if not os.path.exists(dataset_folder_by_people): os.makedirs(dataset_folder_by_people)
    if not os.path.exists(dataset_folder_by_file): os.makedirs(dataset_folder_by_file)

    people_pretext_train_path    = os.path.join(dataset_folder_by_people, "people-pretext_train.csv")
    people_pretext_val_path      = os.path.join(dataset_folder_by_people, "people-pretext_val.csv")
    people_downstream_train_path = os.path.join(dataset_folder_by_people, "people-downstream_train.csv")
    people_downstream_val_path   = os.path.join(dataset_folder_by_people, "people-downstream_val.csv")
    people_downstream_test_path  = os.path.join(dataset_folder_by_people, "people-downstream_test.csv")

    files_pretext_train_path     = os.path.join(dataset_folder_by_file, "files-pretext_train.csv")
    files_pretext_val_path       = os.path.join(dataset_folder_by_file, "files-pretext_val.csv")
    files_downstream_train_path  = os.path.join(dataset_folder_by_file, "files-downstream_train.csv")
    files_downstream_val_path    = os.path.join(dataset_folder_by_file, "files-downstream_val.csv")
    files_downstream_test_path   = os.path.join(dataset_folder_by_file, "files-downstream_test.csv")
   
    print(f"Saving split datasets in {dataset_folder}/...")
    people_pretext_train_df.to_csv(people_pretext_train_path, index=False)
    people_pretext_val_df.to_csv(people_pretext_val_path, index=False)
    people_downstream_train_df.to_csv(people_downstream_train_path, index=False)
    people_downstream_val_df.to_csv(people_downstream_val_path, index=False)
    people_downstream_test_df.to_csv(people_downstream_test_path, index=False)

    files_pretext_train_df.to_csv(files_pretext_train_path, index=False)
    files_pretext_val_df.to_csv(files_pretext_val_path, index=False)
    files_downstream_train_df.to_csv(files_downstream_train_path, index=False)
    files_downstream_val_df.to_csv(files_downstream_val_path, index=False)
    files_downstream_test_df.to_csv(files_downstream_test_path, index=False)

    print("Done!")

# ----------------------------------------------------------------


def display_info_dataset(dataset_path):
    print("Dataset info ------")
    # Read dataset
    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)
    print(dataset_df)

# ----------------------------------------------------------------
# HEAD PREDICTION FUNCTIONS

# Function to load embeddings and labels (MLP)
def load_embeddings_and_labels(embeddings_folder_path, labels_path):
    embeddings = []
    labels = []
    labels_df = pd.read_csv(labels_path)

    for index, row in labels_df.iterrows():
        audio_name = row['file']
        audio_path = os.path.join(embeddings_folder_path, f"{audio_name}_embedding.pt")
        embedding = torch.load(audio_path)
        embeddings.append(embedding)
        labels.append(row['label'])

    return torch.stack(embeddings), torch.tensor(labels)

# Function to load embeddings and labels with NumPy converter (SVM)
def load_embeddings_and_labels_numpy(embeddings_folder_path, labels_path):
    embeddings = []
    labels = []
    labels_df = pd.read_csv(labels_path)

    for index, row in labels_df.iterrows():
        audio_name = row['file']
        audio_path = os.path.join(embeddings_folder_path, f"{audio_name}_embedding.pt")
        embedding = torch.load(audio_path)
        embeddings.append(embedding)
        labels.append(row['label'])

    return torch.stack(embeddings).numpy(), torch.tensor(labels).numpy()