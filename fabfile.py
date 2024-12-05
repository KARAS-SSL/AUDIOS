import json
import os

import optuna
import torch
from fabric import task

from src.models.mlp.test_mlp import test_mlp
from src.models.mlp.train_mlp import train_mlp
from src.models.random_florest.test_rf import test_rf
from src.models.random_florest.train_rf import train_rf
from src.models.svm.test_svm import test_svm
from src.models.svm.train_svm import train_svm
from src.utils.dataset import (
    add_amplitude_dataset,
    add_duration_dataset,
    add_noise_dataset,
    generate_dataset_files_meta,
    generate_dataset_people_meta,
    normalize_dataset,
    split_full_dataset,
)
from src.utils.embeddings import (
    generate_embeddings_hubert,
    generate_embeddings_wav2vec,
    load_embeddings,
)
from src.utils.models import objective_mlp, objective_rf, objective_svm
from src.visualize.tsne import visualize_embeddings_tsne
from src.visualize.umap import visualize_embeddings_umap

# Set the seed value all over the place to make this reproducible
randomness_seed = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# METADATA CSV TASKS
# ------------------------------------------------------------------------------


@task
def GenerateDatasetCSV(c):
    """Generate the dataset metadata .csv files."""

    dataset_path = os.path.join("datasets", "release")
    generate_dataset_people_meta(dataset_path)
    generate_dataset_files_meta(dataset_path)


@task
def AddDatasetDuration(c):
    """Add duration information to the dataset metadata .csv file."""

    dataset_path = os.path.join("datasets", "release", "files-metadata.csv")
    new_dataset_path = os.path.join("datasets", "release", "files-metadata_duration.csv")
    add_duration_dataset(dataset_path, new_dataset_path)


@task
def AddDatasetAmplitude(c):
    """Add amplitude information to the dataset metadata .csv file."""

    dataset_path = os.path.join("datasets", "release", "files-metadata_duration.csv")
    new_dataset_path = os.path.join("datasets", "release", "files-metadata_amplitude.csv")
    add_amplitude_dataset(dataset_path, new_dataset_path)


# ------------------------------------------------------------------------------
# DATA PREPROCESSING TASKS
# ------------------------------------------------------------------------------


@task
def NormalizeDataset(c):
    """Normalize the audios' amplitudes of the dataset."""
    
    dataset_path = os.path.join("datasets", "release", "files-metadata.csv")
    new_dataset_path = os.path.join("datasets", "release")
    normalize_dataset(dataset_path, new_dataset_path)


@task
def AddNoise(c):
    """Add noise to the audios of the dataset."""
    dataset_path = os.path.join("datasets", "release", "files-metadata.csv")
    new_dataset_path = os.path.join("datasets", "release")
    add_noise_dataset(dataset_path, new_dataset_path)


@task
def SplitDataset(c):
    """Split the dataset into pretext (training and validation) and downstream (training, validation and test) sets."""

    people_dataset_path = os.path.join("datasets", "release", "people-metadata.csv")
    files_dataset_path  = os.path.join("datasets", "release", "files-metadata.csv")

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
        randomness_seed,
    )


# ------------------------------------------------------------------------------
# EMBEDDINGS TASKS
# ------------------------------------------------------------------------------


@task
def GenerateEmbeddingsWav2vec2(c):
    """Generate embeddings for the dataset using Wav2Vec2.0."""

    # Dataset
    dataset_folder_path = os.path.join("datasets", "release")
    dataset_meta_path = [
        os.path.join(dataset_folder_path, "splits", "by_file", "files-downstream_train.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file", "files-downstream_val.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file", "files-downstream_test.csv"),
    ]
    sample_rate = 16000

    # Which model to use:
    model_id = "facebook/wav2vec2-base"

    # Embeddings output folder
    if isinstance(dataset_meta_path, str):
        split_name = os.path.basename(dataset_meta_path).split(".")[0]
        embeddings_path = os.path.join("embeddings", model_id.replace("/", "-"), split_name)
        os.makedirs(embeddings_path, exist_ok=True)
    elif isinstance(dataset_meta_path, list):
        split_name = [os.path.basename(path).split(".")[0] for path in dataset_meta_path ]
        embeddings_path = [
            os.path.join("embeddings", model_id.replace("/", "-"), name)
            for name in split_name
        ]
        for path in embeddings_path:
            os.makedirs(path, exist_ok=True)

    generate_embeddings_wav2vec(dataset_folder_path, dataset_meta_path, sample_rate, model_id, embeddings_path)


@task
def GenerateEmbeddingsHubert(c):
    """Generates embeddings for the dataset using HuBERT."""

    # Dataset
    dataset_folder_path = os.path.join("datasets", "release")
    dataset_meta_path = [
        os.path.join(dataset_folder_path, "splits", "by_file", "files-downstream_train.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file", "files-downstream_val.csv"),
        os.path.join(dataset_folder_path, "splits", "by_file", "files-downstream_test.csv"),
    ]
    sample_rate = 16000

    # Which model to use:
    model_id = "facebook/hubert-large-ls960-ft"

    # Embeddings output folder
    if isinstance(dataset_meta_path, str):
        split_name = os.path.basename(dataset_meta_path).split(".")[0]
        embeddings_path = os.path.join("embeddings", model_id.replace("/", "-"), split_name)
        os.makedirs(embeddings_path, exist_ok=True)
    elif isinstance(dataset_meta_path, list):
        split_name = [os.path.basename(path).split(".")[0] for path in dataset_meta_path ]
        embeddings_path = [
            os.path.join("embeddings", model_id.replace("/", "-"), name)
            for name in split_name
        ]
        for path in embeddings_path:
            os.makedirs(path, exist_ok=True)

    generate_embeddings_hubert(dataset_folder_path, dataset_meta_path, sample_rate, model_id, embeddings_path)


@task
def VisualizeEmbeddingsUMAP(c, backbone: str):
    """
    Visualize embeddings using UMAP. Run `fab -h VisualizeEmbeddingsUMAP` to see the list of arguments.

    Args
    ----
    backbone: str
        The backbone used to generate the embeddings. Available options: `wav2vec2`, `hubert`.

    Raises
    ------
    ValueError
        If the backbone is invalid.
    """
    if backbone not in ["wav2vec2", "hubert"]:
        raise ValueError(f"Invalid backbone: {backbone}")

    if backbone == "wav2vec2":
        backbone_folder = "facebook-wav2vec2-base"
    elif backbone == "hubert":
        backbone_folder = "facebook-hubert-large-ls960-ft"

    train_embeddings_folder_path = os.path.join("embeddings", backbone_folder, "files-downstream_train")
    loader = load_embeddings(train_embeddings_folder_path)
    visualize_embeddings_umap(train_embeddings_folder_path, loader, randomness_seed)


@task
def VisualizeEmbeddingsTSNE(c, backbone: str):
    """
    Visualize embeddings using t-SNE. Run `fab -h VisualizeEmbeddingsTSNE` to see the list of arguments.

    Args
    ----
    backbone: str
        The backbone used to generate the embeddings. Available options: `wav2vec2`, `hubert`.

    Raises
    ------
    ValueError
        If the backbone is invalid.
    """
    if backbone not in ["wav2vec2", "hubert"]:
        raise ValueError(f"Invalid backbone: {backbone}")

    if backbone == "wav2vec2":
        backbone_folder = "facebook-wav2vec2-base"
    elif backbone == "hubert":
        backbone_folder = "facebook-hubert-large-ls960-ft"

    train_embeddings_folder_path = os.path.join("embeddings", backbone_folder, "files-downstream_train")
    loader = load_embeddings(train_embeddings_folder_path)
    visualize_embeddings_tsne(train_embeddings_folder_path, loader, randomness_seed)


@task
def VisualizeEmbeddings(c, backbone: str):
    """
    Visualize embeddings using UMAP and t-SNE. Run `fab -h VisualizeEmbeddings` to see the list of arguments.

    Args
    ----
    backbone: str
        The backbone used to generate the embeddings. Available options: `wav2vec2`, `hubert`.

    Raises
    ------
    ValueError
        If the backbone is invalid.
    """
    if backbone not in ["wav2vec2", "hubert"]:
        raise ValueError(f"Invalid backbone: {backbone}")

    if backbone == "wav2vec2":
        backbone_folder = "facebook-wav2vec2-base"
    elif backbone == "hubert":
        backbone_folder = "facebook-hubert-large-ls960-ft"

    train_embeddings_folder_path = os.path.join("embeddings", backbone_folder, "files-downstream_train")
    loader = load_embeddings(train_embeddings_folder_path)
    visualize_embeddings_umap(train_embeddings_folder_path, loader, randomness_seed)
    visualize_embeddings_tsne(train_embeddings_folder_path, loader, randomness_seed)


# ----------------------------------------------------------------------------
# MODEL TRAINING TASKS
# ----------------------------------------------------------------------------


@task
def OptimizeHyperparameters(c, backbone: str, classifier: str):
    """
    Execute a study to optimize the hyperparameters for the given classifier and embeddings. Run `fab -h OptimizeHyperparameters` to see the list of arguments.
    
    Args
    ----
    backbone: str
        The backbone used to generate the embeddings. Available options: `wav2vec2`, `hubert`.
    classifier: str
        The classifier to use for training. Available options: `mlp`, `svm`, `rf`.
    
    Raises
    ------
    ValueError
        If the backbone or classifier is invalid.
    """
    if backbone not in ["wav2vec2", "hubert"]:
        raise ValueError(f"Invalid backbone: {backbone}")
    if classifier not in ["mlp", "svm", "rf"]:
        raise ValueError(f"Invalid prediction head: {classifier}")

    # Define the model folder
    if backbone == "wav2vec2":
        backbone_folder = "facebook-wav2vec2-base"
    elif backbone == "hubert":
        backbone_folder = "facebook-hubert-large-ls960-ft"

    # Load embeddings
    train_embeddings_folder_path = os.path.join("embeddings", backbone_folder, "files-downstream_train")
    val_embeddings_folder_path   = os.path.join("embeddings", backbone_folder, "files-downstream_val")
    train_embeddings_loader = load_embeddings(train_embeddings_folder_path)
    val_embeddings_loader   = load_embeddings(val_embeddings_folder_path)

    # Create output folder for the study runs
    output_path = "runs"
    os.makedirs(output_path, exist_ok=True)
    study_number = len(
        [f for f in os.listdir(output_path) if f.startswith(f"{classifier}_study")]
    )
    study_folder = os.path.join(output_path, f"{classifier}_study{study_number}")
    os.makedirs(study_folder, exist_ok=True)

    # Optimize the hyperparameters
    study = optuna.create_study(direction="minimize")
    if classifier == "mlp":
        study.optimize(
            lambda trial: objective_mlp(
                trial,
                train_embeddings_loader,
                val_embeddings_loader,
                train_embeddings_folder_path,
                val_embeddings_folder_path,
                study_folder,
            ),
            n_trials=100,
        )
    elif classifier == "svm":
        study.optimize(
            lambda trial: objective_svm(
                trial,
                train_embeddings_loader,
                val_embeddings_loader,
                train_embeddings_folder_path,
                val_embeddings_folder_path,
                study_folder,
            ),
            n_trials=30,
        )
    elif classifier == "rf":
        study.optimize(
            lambda trial: objective_rf(
                trial,
                train_embeddings_loader,
                val_embeddings_loader,
                train_embeddings_folder_path,
                val_embeddings_folder_path,
                study_folder,
            ),
            n_trials=30,
        )

    # Save the best hyperparameters to a file
    best_run_path = os.path.join(study_folder, "best_hyperparameters.json")
    best_run = {
        "best_trial_number": study.best_trial.number,
        "best_validation_eer": study.best_value,
        "best_hyperparameters": study.best_params,
    }
    with open(best_run_path, "w") as f:
        json.dump(best_run, f, indent=4)

    print("Study completed. Best hyperparameters saved to ", best_run_path)


@task
def TrainModel(c, backbone: str, classifier: str):
    """
    Train a classifier using the embeddings from the given backbone and predefined hyperparameters. Run `fab -h TrainModel` to see the list of arguments.
    
    Args
    ----
    backbone: str
        The backbone used to generate the embeddings. Available options: `wav2vec2`, `hubert`.
    classifier: str
        The classifier to use for training. Available options: `mlp`, `svm`, `rf`.
    
    Raises
    ------
    ValueError
        If the backbone or classifier is invalid.
    """
    if backbone not in ["wav2vec2", "hubert"]:
        raise ValueError(f"Invalid backbone: {backbone}")
    if classifier not in ["mlp", "svm", "rf"]:
        raise ValueError(f"Invalid prediction head: {classifier}")
    
    # Define the model folder
    if backbone == "wav2vec2":
        backbone_folder = "facebook-wav2vec2-base"
    elif backbone == "hubert":
        backbone_folder = "facebook-hubert-large-ls960-ft"

    # Load embeddings
    train_embeddings_folder_path = os.path.join("embeddings", backbone_folder, "files-downstream_train")
    val_embeddings_folder_path   = os.path.join("embeddings", backbone_folder, "files-downstream_val")
    train_embeddings_loader = load_embeddings(train_embeddings_folder_path)
    val_embeddings_loader   = load_embeddings(val_embeddings_folder_path)

    output_path = "runs"

    if classifier == "mlp":
        hyperparameters = {
            "epochs": 50,
            "batch_size": 128,
            "learning_rate": 3.32e-05,
            "patience": 10,
            "dropout": 0.38,
            "weight_decay": 0.03,
            "hidden_dim_1": 256,
            "output_dim": 1,
            "randomness_seed": randomness_seed,
        }
        train_mlp(
            train_embeddings_loader,
            val_embeddings_loader,
            train_embeddings_folder_path,
            val_embeddings_folder_path,
            hyperparameters,
            output_path,
            randomness_seed,
            device,
        )
    elif classifier == "svm":
        hyperparameters = {
            "batch_size": 64,
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "randomness_seed": randomness_seed,
        }
        train_svm(
            train_embeddings_loader,
            val_embeddings_loader,
            train_embeddings_folder_path,
            val_embeddings_folder_path,
            hyperparameters,
            output_path,
            randomness_seed,
        )
    elif classifier == "rf":
        hyperparameters = {
            "batch_size": 64,
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "randomness_seed": randomness_seed,
        }
        train_rf(
            train_embeddings_loader,
            val_embeddings_loader,
            train_embeddings_folder_path,
            val_embeddings_folder_path,
            hyperparameters,
            output_path,
            randomness_seed,
        )


# -------------------------------------------------------------
# MODEL TESTING TASKS
# -------------------------------------------------------------


@task
def TestModel(c, backbone: str, classifier: str, run_number: int, study_number: int | None = None, gender: str = ""):
    """
    Test a classifier using the embeddings from the given backbone. Run `fab -h TestModel` to see the list of arguments.

    Args
    ----
    backbone: str
        The backbone used to generate the embeddings. Available options: `wav2vec2`, `hubert`.
    classifier: str
        The classifier to use for testing. Available options: `mlp`, `svm`, `rf`.
    run-number: int
        The number of the run to test.
    study-number: int, optional
        The number of the study to test. If not provided, will assume the run is not part of a study. Default is None.
    gender: str, optional
        Test only embeddings of the specified gender. Available options: `M`, `F` or empty (both genders). Default is empty.

    Raises
    ------
    ValueError
        If the backbone, classifier or gender is invalid.
    """
    if backbone not in ["wav2vec2", "hubert"]:
        raise ValueError(f"Invalid backbone: {backbone}")
    if classifier not in ["mlp", "svm", "rf"]:
        raise ValueError(f"Invalid prediction head: {classifier}")
    if gender not in ["M", "F", ""]:
        raise ValueError(f"Invalid gender: {gender}")

    # Load embeddings
    if backbone == "wav2vec2":
        backbone_folder = "facebook-wav2vec2-base"
    elif backbone == "hubert":
        backbone_folder = "facebook-hubert-large-ls960-ft"

    test_embeddings_folder_path = os.path.join("embeddings", backbone_folder, "files-downstream_test")

    # Load model
    if study_number is not None:
        model_folder = os.path.join("runs", f"{classifier}_study{study_number}", f"run{run_number}")
    else:
        model_folder = os.path.join("runs", f"run{run_number}")

    if classifier == "mlp":
        test_mlp(test_embeddings_folder_path, model_folder, gender, use_best_model=True, device=device)
    elif classifier == "svm":
        test_svm(test_embeddings_folder_path, model_folder, gender)
    elif classifier == "rf":
        test_rf(test_embeddings_folder_path, model_folder, gender)
