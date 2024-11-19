
import os
from fabric import task

from src.preprocess.dataset_utils    import add_duration_dataset, add_amplitude_dataset, balance_dataset, display_info_dataset, generate_dataset_meta, normalize_dataset
from src.preprocess.embeddings_utils import generate_embeddings_wav2vec

# Set the seed value all over the place to make this reproducible
randomness_seed = 7

#------------------------------------------------------------------------------

@task
def GenerateDatasetCSV(c):
    """Generates the dataset .csv file."""
    
    dataset_path = "datasets/release/"
    generate_dataset_meta(dataset_path)

@task
def AddDatasetDuration(c):
    """Adds duration information to the dataset."""
    
    dataset_path     = "datasets/release/meta.csv"
    new_dataset_path = "datasets/release/meta_duration.csv" 
    add_duration_dataset(dataset_path, new_dataset_path)

@task
def AddDatasetAmplitude(c):
    """Adds amplitude information to the dataset."""
    
    dataset_path     = "datasets/release/meta.csv"
    new_dataset_path = "datasets/release/meta_amplitude.csv" 
    add_amplitude_dataset(dataset_path, new_dataset_path)

@task
def NormalizeDataset(c):
    """Normalizes the audio amplitudes of the dataset."""
    
    dataset_path     = "datasets/release/meta.csv"
    new_dataset_path = "datasets/release/meta_normalized.csv" 
    normalize_dataset(dataset_path, new_dataset_path)

@task
def BalanceDataset(c):
    """Balances the dataset."""
   
    # Set threshold k for allowable imbalance (e.g., 0.2 allows 20-80% balance)
    # For each person in the dataset, if the spoof ratio is less than k or greater than (1 - k) then discard
    # the samples of the biggest label for that person such that spoof == bonafide
    imbalance_threshold = 0.3
    
    dataset_path        = "datasets/release_in_the_wild/meta_duration.csv"
    new_dataset_path    = "datasets/release_in_the_wild/meta_balanced.csv" 
    balance_dataset(dataset_path, new_dataset_path, imbalance_threshold, randomness_seed)

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
    dataset_path      = "datasets/release_in_the_wild/meta_balanced.csv"
    sample_rate       = 16000
    
    # Which model to use:
    model_id_wav2vec  = "facebook/wav2vec2-base-960h"
    model_id_xlsr     = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    model_id          = model_id_xlsr

    # Embeddings output folder
    embeddings_path = "embeddings/wav2vec/in_the_wild__" + model_id.replace("/", "-")
    os.makedirs(embeddings_path, exist_ok=True)
     
    generate_embeddings_wav2vec(dataset_path, sample_rate, model_id, embeddings_path)

