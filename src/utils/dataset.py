import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

tqdm.pandas()


# ----------------------------------------------------------------
# AUDIO FILE PROPERTIES FUNCTIONS

# Function to load an audio file
def load_audio_file(filename: str, dataset_folder_path: str) -> np.ndarray | None:
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
    files = {}

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
            audio_path = os.path.join("fake_voices", folder, filename)  # relative path inside the dataset
            # files.append([audio_path, speaker_name, speaker_id, speaker_gender, "spoof"])
            files[audio_path] = {
                "speaker": speaker_name,
                "id": speaker_id,
                "gender": speaker_gender,
                "label": "spoof"
            }

    # Process real audios
    for folder in os.listdir(real_audios_path):
        folder_path = os.path.join(real_audios_path, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directories

        try:
            speaker_name, speaker_id, *_ = folder.split("_")
            speaker_gender = speaker_id[0]
        except ValueError:
            print(f"Skipping folder with unexpected format: {folder}")
            continue

        for filename in os.listdir(folder_path):
            audio_path = os.path.join("real_voices", folder, filename)
            # files.append([audio_path, speaker_name, speaker_id, speaker_gender, "bona-fide"])
            files[audio_path] = {
                "speaker": speaker_name,
                "id": speaker_id,
                "gender": speaker_gender,
                "label": "bona-fide"
            }

    # Write metadata to CSV
    fields = ["file", "speaker", "id", "gender", "label"]
    output_file = os.path.join(dataset_folder_path, "files-metadata.csv")
    pd.DataFrame(files, columns=fields).to_csv(output_file, index=False)
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

    # Write metadata to CSV
    fields = ["person", "gender", "id", "spoof_count", "bonafide_count", "spoof_folder", "bonafide_folder"]
    output_file = os.path.join(dataset_folder_path, "people-metadata.csv")
    pd.DataFrame(people, columns=fields).to_csv(output_file, index=False)
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

    print("Dataset normalized. New dataset saved to ", new_dataset_folder_path)


# ----------------------------------------------------------------
# Balances the dataset by removing samples with low duration
def balance_dataset(dataset_path, new_dataset_path, imbalance_threshold, seed):
    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)

    # Per speaker counts
    spoof_per_speaker = (
        dataset_df[dataset_df.label == "spoof"].groupby("speaker").duration.count()
    )
    bonafide_per_speaker = (
        dataset_df[dataset_df.label == "bona-fide"].groupby("speaker").duration.count()
    )

    # Combine counts into a single DataFrame and fill NaN with 0
    counts_df = pd.DataFrame(
        {"spoof_count": spoof_per_speaker, "bona_fide_count": bonafide_per_speaker}
    ).fillna(0)

    # Calculate total count and spoof ratio for each speaker
    counts_df["total"] = counts_df["spoof_count"] + counts_df["bona_fide_count"]
    counts_df["spoof_ratio"] = counts_df["spoof_count"] / counts_df["total"]

    # Initialize list to hold balanced data
    final_data = []

    for speaker in counts_df.index:
        speaker_data = dataset_df[dataset_df["speaker"] == speaker]

        # Separate spoof and bonafide samples
        spoof_samples = speaker_data[speaker_data["label"] == "spoof"]
        bonafide_samples = speaker_data[speaker_data["label"] == "bona-fide"]

        # Calculate the spoof ratio for this speaker
        spoof_ratio = counts_df.loc[speaker, "spoof_ratio"]

        # Check if the speaker's spoof/bona-fide ratio is within the threshold
        if spoof_ratio < imbalance_threshold or spoof_ratio > (1 - imbalance_threshold):
            # Balance spoof and bona-fide samples by truncating the excess
            target_count = min(len(spoof_samples), len(bonafide_samples))
            spoof_samples = spoof_samples.sample(n=target_count, random_state=seed)
            bonafide_samples = bonafide_samples.sample(
                n=target_count, random_state=seed
            )

        # Append both spoof and bona-fide samples for this speaker
        final_data.append(pd.concat([spoof_samples, bonafide_samples]))

    # Concatenate all balanced samples and sort by numeric part of 'file' column
    balanced_dataset_df = pd.concat(final_data)
    balanced_dataset_df["file_index"] = (
        balanced_dataset_df["file"].str.extract(r"(\d+)").astype(int)
    )
    balanced_dataset_df = (
        balanced_dataset_df.sort_values(by="file_index")
        .drop(columns="file_index")
        .reset_index(drop=True)
    )

    balanced_dataset_df.to_csv(new_dataset_path, index=False)
    print("Balanced dataset saved to", new_dataset_path)


# ----------------------------------------------------------------


def display_info_dataset(dataset_path):
    print("Dataset info ------")
    # Read dataset
    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)
    print(dataset_df)

    # Dataset duration
    # print(dataset_df.groupby("label")['duration'].sum().sort_values(ascending=False) / 3600, end="\n\n")

    # Spoof and bonafide per speaker
    # spoof_per_speaker    = dataset_df[dataset_df.label == 'spoof'].groupby("speaker").duration.count()
    # bonafide_per_speaker = dataset_df[dataset_df.label == 'bona-fide'].groupby("speaker").duration.count()
    # counts_df = pd.DataFrame({
    #     'spoof_count': spoof_per_speaker,
    #     'bona_fide_count': bonafide_per_speaker
    # }).fillna(0)
    # counts_df['total']       = counts_df['spoof_count'] + counts_df['bona_fide_count']
    # counts_df['spoof_ratio'] = counts_df['spoof_count'] / counts_df['total']
    # counts_df.sort_values("spoof_ratio")
    # print(counts_df)
