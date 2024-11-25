
import json
import os
from fabric import task

from src.utils.dataset    import add_duration_dataset, add_amplitude_dataset, display_info_dataset, generate_dataset_files_meta, generate_dataset_people_meta, normalize_dataset, split_full_dataset

from src.utils.embeddings import generate_embeddings_wav2vec, generate_embeddings_wav2vec2_bert, generate_embeddings_hubert

from src.train.mlp.train_mlp import train_mlp 
from src.train.mlp.test_mlp import test_mlp 

from src.visualize.umap import visualize_embeddings_umap
from src.visualize.tsne import visualize_embeddings_tsne

import optuna
import torch
import numpy as np

# Set the seed value all over the place to make this reproducible
randomness_seed = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def NormalizeDataset(c, use_same_dir=False):
    """Normalizes the audio amplitudes of the dataset."""
    
    dataset_path     = "datasets/release/files-metadata.csv"
    new_dataset_path = "datasets/release/" if use_same_dir else "datasets/normalized/"
    normalize_dataset(dataset_path, new_dataset_path)

@task
def SplitDataset(c):
    """Splits the dataset into pretext (training and validation) and downstream (training, validation and test) sets."""

    people_dataset_path = "datasets/release/people-metadata.csv"
    files_dataset_path  = "datasets/release/files-metadata.csv"

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
    dataset_path = "datasets/release/files-metadata.csv" 
    display_info_dataset(dataset_path)

#----------------------------------------------------------------------------

@task
def GenerateEmbeddingsWav2vec2(c):
    """Generates embeddings for the dataset using Wav2vec."""

    # Dataset
    dataset_folder_path = "datasets/release"
    dataset_meta_path   = [
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_train.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_val.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_test.csv")    
    ]
    sample_rate         = 16000

    # Which model to use:
    model_id = "facebook/wav2vec2-base-960h"

    # Embeddings output folder
    if isinstance(dataset_meta_path, str):
        split_name = os.path.basename(dataset_meta_path).split(".")[0] 
        embeddings_path = os.path.join("embeddings", model_id.replace("/", "-"), split_name)
        os.makedirs(embeddings_path, exist_ok=True)
    elif isinstance(dataset_meta_path, list):
        split_name = [os.path.basename(path).split(".")[0] for path in dataset_meta_path]
        embeddings_path = [os.path.join("embeddings", model_id.replace("/", "-"), name) for name in split_name]
        for path in embeddings_path:
            os.makedirs(path, exist_ok=True)

    generate_embeddings_wav2vec(dataset_folder_path, dataset_meta_path, sample_rate, model_id, embeddings_path)

@task
def GenerateEmbeddingsWav2vec2BERT(c):
    """Generates embeddings for the dataset using Wav2vec."""

    # Dataset
    dataset_folder_path = "datasets/release"
    dataset_meta_path   = [
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_train.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_val.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_test.csv")    
    ]
    sample_rate         = 16000

    # Which model to use:
    model_id = "hf-audio/wav2vec2-bert-CV16-en"

    # Embeddings output folder
    if isinstance(dataset_meta_path, str):
        split_name = os.path.basename(dataset_meta_path).split(".")[0] 
        embeddings_path = os.path.join("embeddings", model_id.replace("/", "-"), split_name)
        os.makedirs(embeddings_path, exist_ok=True)
    elif isinstance(dataset_meta_path, list):
        split_name = [os.path.basename(path).split(".")[0] for path in dataset_meta_path]
        embeddings_path = [os.path.join("embeddings", model_id.replace("/", "-"), name) for name in split_name]
        for path in embeddings_path:
            os.makedirs(path, exist_ok=True)

    generate_embeddings_wav2vec2_bert(dataset_folder_path, dataset_meta_path, sample_rate, model_id, embeddings_path)

    
@task
def GenerateEmbeddingsHubert(c):
    """Generates embeddings for the dataset using HuBERT."""

    # Dataset
    dataset_folder_path = "datasets/release"
    dataset_meta_path   = [
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_train.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_val.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file" ,"files-downstream_test.csv")    
    ] 
    sample_rate         = 16000

    # Which model to use:
    model_id = "facebook/hubert-large-ls960-ft"

    # Embeddings output folder
    if isinstance(dataset_meta_path, str):
        split_name = os.path.basename(dataset_meta_path).split(".")[0] 
        embeddings_path = os.path.join("embeddings", model_id.replace("/", "-"), split_name)
        os.makedirs(embeddings_path, exist_ok=True)
    elif isinstance(dataset_meta_path, list):
        split_name = [os.path.basename(path).split(".")[0] for path in dataset_meta_path]
        embeddings_path = [os.path.join("embeddings", model_id.replace("/", "-"), name) for name in split_name]
        for path in embeddings_path:
            os.makedirs(path, exist_ok=True)

    generate_embeddings_hubert(dataset_folder_path, dataset_meta_path, sample_rate, model_id, embeddings_path)

#----------------------------------------------------------------------------
# Visualize Embeddings

@task
def VisualizeEmbeddingsUMAP(c):
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base-960h/files-downstream_train"
    visualize_embeddings_umap(train_embeddings_folder_path) 
    pass

@task
def VisualizeEmbeddingsTSNE(c):
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base-960h/files-downstream_train" 
    visualize_embeddings_tsne(train_embeddings_folder_path) 
    pass
    
#----------------------------------------------------------------------------
# Train Model

def objective_mlp(trial, output_path):
    # Define the hyperparameter search space
    hyperparameters = {
        "epochs": 50,
        "patience": 10,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "hidden_dim_1": trial.suggest_categorical("hidden_dim_1", [128, 256, 512]),
        "output_dim": 1
    }

    # Paths to embeddings
    train_embeddings_folder_path = "embeddings/facebook-hubert-large-ls960-ft/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-hubert-large-ls960-ft/files-downstream_val"
    
    # Call the training function
    validation_eer = train_mlp(
        train_embeddings_folder_path, 
        val_embeddings_folder_path, 
        hyperparameters, 
        output_path, 
        randomness_seed, 
        device
    )

    # Return the validation EER as the objective metric
    return validation_eer


@task
def OptimizeHyperparameters(c, prediction_head: str):

    if prediction_head != "mlp" and prediction_head != "svm":
        raise ValueError(f"Invalid prediction head: {prediction_head}")

    # Create output folder for the study runs
    output_path = "runs"
    os.makedirs(output_path, exist_ok=True)
    study_number = len([f for f in os.listdir(output_path) if f.startswith(f"{prediction_head}_study")])
    study_folder = os.path.join(output_path, f"{prediction_head}_study{study_number}")
    os.makedirs(study_folder, exist_ok=True)

    # Optimize the hyperparameters
    study = optuna.create_study(direction="minimize")
    if prediction_head == "mlp":
        study.optimize(lambda trial: objective_mlp(trial, study_folder), n_trials=5)
    elif prediction_head == "svm":
        print("Not implemented yet. :P")

    # Save the best hyperparameters to a file
    best_run_path = os.path.join(study_folder, "best_hyperparameters.json")
    best_run      = {
        "best_trial_number": study.best_trial.number,
        "best_validation_eer": study.best_value,
        "best_hyperparameters": study.best_params
    }
    with open(best_run_path, "w") as f:
        json.dump(best_run, f, indent=4)
    
    print("Study completed. Best hyperparameters saved to ", best_run_path)


@task
def TrainModel(c):
    
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base-960h/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base-960h/files-downstream_val"

    prediction_head = "mlp"
    output_path     = "runs"

    if prediction_head == "mlp":
        hyperparameters = {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "patience": 10,
            "dropout": 0.4,
            "weight_decay": 0.01,
            "hidden_dim_1": 256,
            "output_dim": 1 
        }
        train_mlp(train_embeddings_folder_path, val_embeddings_folder_path, hyperparameters, output_path, randomness_seed, device)
    elif prediction_head == "svm":
        print("Not implemented yet. :P")


@task
def TestModel(c):
    
    test_embeddings_folder_path   = "embeddings/facebook-hubert-large-ls960-ft/files-downstream_test"
    best_hyperparameters_path     = "runs/mlp_study1"
    prediction_head               = "mlp"

    with open(os.path.join(best_hyperparameters_path, "best_hyperparameters.json"), "r") as f:
        best_hyperparameters = json.load(f)
        best_model_idx       = best_hyperparameters.get("best_trial_number")
    
    model_folder = os.path.join(best_hyperparameters_path, f"run{best_model_idx}")

    if prediction_head == "mlp":
        test_mlp(test_embeddings_folder_path, model_folder, use_best_model=True, device=device)
    elif prediction_head == "svm":
        print("Not implemented yet. :P")
        
#----------------------------------------------------------------------------
# Test Model

    
