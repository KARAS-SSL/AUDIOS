
import json
import os
from fabric import task

from src.utils.dataset import add_duration_dataset, add_amplitude_dataset, generate_dataset_files_meta, generate_dataset_people_meta, normalize_dataset, split_full_dataset, add_noise_dataset, load_embeddings

from src.utils.embeddings import generate_embeddings_wav2vec, generate_embeddings_hubert

from src.models.mlp.train_mlp import train_mlp 
from src.models.mlp.test_mlp import test_mlp 

from src.models.svm.train_svm import train_svm
from src.models.svm.test_svm import test_svm

from src.models.random_florest.train_rf import train_rf
from src.models.random_florest.test_rf import test_rf

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
def AddNoise(c, use_same_dir=False):
    dataset_path     = "datasets/release/files-metadata.csv" 
    new_dataset_path = "datasets/release/"
    add_noise_dataset(dataset_path, new_dataset_path)
    
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

# @task
# def DisplayDatasetInfo(c):
#     """Displays information about the dataset.""" 
#     dataset_path = "datasets/release/files-metadata.csv" 
#     display_info_dataset(dataset_path)

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
    model_id = "facebook/wav2vec2-base"

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
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train/"
    loader = load_embeddings(train_embeddings_folder_path)
    visualize_embeddings_umap(loader) 

@task
def VisualizeEmbeddingsTSNE(c):
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train/" 
    loader = load_embeddings(train_embeddings_folder_path)
    visualize_embeddings_tsne(loader) 

@task 
def VisualizeEmbeddings(c):
    train_embeddings_folder_path = "embeddings/facebook-hubert-large-ls960-ft/files-downstream_train/"
    loader = load_embeddings(train_embeddings_folder_path)
    visualize_embeddings_umap(loader)
    visualize_embeddings_tsne(loader)
    
#----------------------------------------------------------------------------
# Train Model

def objective_mlp(trial, train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, output_path):
    # Define the hyperparameter search space
    hyperparameters = {
        "epochs": 50,
        "patience": 8,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e1, log=True),
        "hidden_dim_1": trial.suggest_categorical("hidden_dim_1", [128, 256, 512]),
        "output_dim": 1,
        "randomness_seed": randomness_seed
    }

    # Paths to embeddings   
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base/files-downstream_val"
    
    # Call the training function
    validation_eer = train_mlp(
        train_embeddings_loader, 
        val_embeddings_loader,
        train_embeddings_folder_path, 
        val_embeddings_folder_path, 
        hyperparameters, 
        output_path, 
        randomness_seed, 
        device
    )

    # Return the validation EER as the objective metric
    return validation_eer


def objective_svm(trial, train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, output_path):
    # Define the hyperparameter search space
    hyperparameters = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "C": trial.suggest_float("C", 0.1, 10.0),
        "kernel": "rbf",
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "randomness_seed": randomness_seed

    }

    # Paths to embeddings
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base/files-downstream_val"

 
    # Call the training function
    validation_eer = train_svm(
        train_embeddings_loader, 
        val_embeddings_loader,
        train_embeddings_folder_path, 
        val_embeddings_folder_path, 
        hyperparameters, 
        output_path, 
        randomness_seed
    )

    # Return the validation EER as the objective metric
    return validation_eer


def objective_rf(trial, train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, output_path):
    # Define the hyperparameter search space
    hyperparameters = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "randomness_seed": randomness_seed

    }

    # Paths to embeddings
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base/files-downstream_val"
 
    # Call the training function
    validation_eer = train_rf(
        train_embeddings_loader, 
        val_embeddings_loader,
        train_embeddings_folder_path, 
        val_embeddings_folder_path, 
        hyperparameters, 
        output_path, 
        randomness_seed
    )

    # Return the validation EER as the objective metric
    return validation_eer


@task
def OptimizeHyperparameters(c, prediction_head: str):

    if prediction_head != "mlp" and prediction_head != "svm" and prediction_head != "rf":
        raise ValueError(f"Invalid prediction head: {prediction_head}")

    # Load embeddings
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base/files-downstream_val"
    train_embeddings_loader = load_embeddings(train_embeddings_folder_path)
    val_embeddings_loader   = load_embeddings(val_embeddings_folder_path)

    # Create output folder for the study runs
    output_path = "runs"
    os.makedirs(output_path, exist_ok=True)
    study_number = len([f for f in os.listdir(output_path) if f.startswith(f"{prediction_head}_study")])
    study_folder = os.path.join(output_path, f"{prediction_head}_study{study_number}")
    os.makedirs(study_folder, exist_ok=True)

    # Optimize the hyperparameters
    study = optuna.create_study(direction="minimize")
    if prediction_head == "mlp":
        study.optimize(
            lambda trial: objective_mlp(trial, train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, study_folder),
            n_trials=100
        )
    elif prediction_head == "svm":
        study.optimize(
            lambda trial: objective_svm(trial, train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, study_folder),
            n_trials=30
        )
    elif prediction_head == "rf":
        study.optimize(
            lambda trial: objective_rf(trial, train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, study_folder),
            n_trials=30
        )

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
    train_embeddings_folder_path = "embeddings/facebook-wav2vec2-base/files-downstream_train"
    val_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base/files-downstream_val"
    train_embeddings_loader = load_embeddings(train_embeddings_folder_path)
    val_embeddings_loader   = load_embeddings(val_embeddings_folder_path)

    prediction_head = "mlp"
    output_path     = "runs"

    if prediction_head == "mlp":
        hyperparameters = {
            "epochs": 50,
            "batch_size": 128,
            "learning_rate": 3.32e-05,
            "patience": 10,
            "dropout": 0.38,
            "weight_decay": 0.03,
            "hidden_dim_1": 256,
            "output_dim": 1,
            "randomness_seed": randomness_seed
        }
        train_mlp(train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, hyperparameters, output_path, randomness_seed, device)
    elif prediction_head == "svm":
        hyperparameters = {
            "batch_size": 64, 
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "randomness_seed": randomness_seed
        }
        train_svm(train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, hyperparameters, output_path, randomness_seed)
    elif prediction_head == "rf":
        hyperparameters = {
            "batch_size": 64, 
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "randomness_seed": randomness_seed
        }
        train_rf(train_embeddings_loader, val_embeddings_loader, train_embeddings_folder_path, val_embeddings_folder_path, hyperparameters, output_path, randomness_seed)
   
@task
def TestModel(c):
    
    test_embeddings_folder_path   = "embeddings/facebook-wav2vec2-base/files-downstream_test" 
    # best_hyperparameters_path     = "runs/run4"
    # prediction_head               = "mlp"
    gender                        = ""

    # with open(os.path.join(best_hyperparameters_path, "best_hyperparameters.json"), "r") as f:
    #     best_hyperparameters = json.load(f)
    #     best_model_idx       = best_hyperparameters.get("best_trial_number")
        
    model_folder                  = "runs/mlp_study3/run22"
    # model_folder                  = "runs/run4"
    prediction_head               = "mlp" 

    # model_folder                  = "runs/run5"
    
    # model_folder = os.path.join(best_hyperparameters_path, f"run{best_model_idx}")

    if prediction_head == "mlp":
        test_mlp(test_embeddings_folder_path, model_folder, use_best_model=True, device=device)
    elif prediction_head == "svm":
        test_svm(test_embeddings_folder_path, model_folder, gender)
    elif prediction_head == "rf":
        test_rf(test_embeddings_folder_path, model_folder)
        
#----------------------------------------------------------------------------
# Test Model
  
