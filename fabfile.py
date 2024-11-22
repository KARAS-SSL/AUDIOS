
import os
from fabric import task

from src.utils.dataset    import add_duration_dataset, add_amplitude_dataset, display_info_dataset, generate_dataset_files_meta, generate_dataset_people_meta, normalize_dataset, split_full_dataset
from src.utils.embeddings import generate_embeddings_wav2vec

# Set the seed value all over the place to make this reproducible
randomness_seed = 7

#------------------------------------------------------------------------------

@task
def GenerateDatasetCSV(c):
    """Generates the dataset metadata .csv files."""
    
    dataset_path = "datasets/release/"
    generate_dataset_people_meta(dataset_path)
    generate_dataset_files_meta(dataset_path)

@task
def AddDatasetDuration(c):
    """Adds duration information to the dataset."""
    
    dataset_path     = "datasets/release/files-metadata.csv"
    new_dataset_path = "datasets/release/files-metadata_duration.csv" 
    add_duration_dataset(dataset_path, new_dataset_path)

@task
def AddDatasetAmplitude(c):
    """Adds amplitude information to the dataset."""
    
    dataset_path     = "datasets/release/files-metadata_duration.csv"
    new_dataset_path = "datasets/release/files-metadata_duration_amplitude.csv" 
    add_amplitude_dataset(dataset_path, new_dataset_path)

@task
def NormalizeDataset(c):
    """Normalizes the audio amplitudes of the dataset."""
    
    dataset_path     = "datasets/release/files-metadata.csv"
    new_dataset_path = "datasets/normalized/" 
    normalize_dataset(dataset_path, new_dataset_path)

@task
def SplitDataset(c):
    """Splits the dataset into pretext (training and validation) and downstream (training, validation and test) sets."""

    people_dataset_path = "datasets/normalized/people-metadata.csv"
    files_dataset_path  = "datasets/normalized/files-metadata.csv"

    # Full dataset: 60% pretrain, 40% downstream
    pretext_percentage    = 0.6
    downstream_percentage = 0.4

    # Pretext: 80% train, 20% val
    pretext_train_percentage = pretext_percentage * 0.8
    pretext_val_percentage   = pretext_percentage * 0.2

    # Downstream: 70% train, 20% val, 10% test
    downstream_train_percentage = downstream_percentage * 0.7
    downstream_val_percentage   = downstream_percentage * 0.2
    downstream_test_percentage  = downstream_percentage * 0.1

    split_full_dataset(
        people_dataset_path,
        files_dataset_path,
        pretext_train_percentage,
        pretext_val_percentage,
        downstream_train_percentage,
        downstream_val_percentage,
        downstream_test_percentage,
        randomness_seed
    )

#------------------------------------------------------------------------------

@task
def DisplayDatasetInfo(c):
    """Displays information about the dataset.""" 
    dataset_path = "datasets/release/metadata.csv" 
    display_info_dataset(dataset_path)

#----------------------------------------------------------------------------

@task
def GenerateEmbeddingsWav2vec2(c):
    """Generates embeddings for the dataset using Wav2vec."""

    # Dataset
    dataset_path      = ["datasets/normalized/files-downstream_val copy.csv", "datasets/normalized/files-downstream_test copy.csv"]
    sample_rate       = 16000

    # Which model to use:
    model_id_wav2vec  = "facebook/wav2vec2-base-960h"
    model_id_xlsr     = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    model_id          = model_id_xlsr

    # Embeddings output folder
    embeddings_path = "embeddings/wav2vec/" + model_id.replace("/", "-")

    generate_embeddings_wav2vec(dataset_path, sample_rate, model_id, embeddings_path)
