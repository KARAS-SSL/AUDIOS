from src.models.mlp.train_mlp import train_mlp
from src.models.random_florest.train_rf import train_rf
from src.models.svm.train_svm import train_svm


# ----------------------------------------------------------------
# Objective functions for hyperparameter tuning

def objective_mlp(
    trial,
    train_embeddings_loader: list,
    val_embeddings_loader: list,
    train_embeddings_folder_path: str,
    val_embeddings_folder_path: str,
    output_path: str,
    randomness_seed: int,
    device: str,
) -> float:
    """
    Objective function for hyperparameter tuning of a Multi-Layer Perceptron (MLP) model.

    Parameters
    ----------
    trial : optuna.Trial
        The trial object from Optuna.
    train_embeddings_loader : list
        A list of tuples containing the embeddings and labels for the training data.
    val_embeddings_loader : list
        A list of tuples containing the embeddings and labels for the validation data.
    train_embeddings_folder_path : str
        The path to the folder containing the embeddings for the training data.
    val_embeddings_folder_path : str
        The path to the folder containing the embeddings for the validation data.
    output_path : str
        The path to save the model to.
    randomness_seed : int
        The seed for random number generation.
    device : str
        The device to use for training (e.g., 'cpu' or 'cuda').

    Returns
    -------
    float
        The validation EER as the objective metric.
    """
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
        "randomness_seed": randomness_seed,
    }

    # Call the training function
    validation_eer = train_mlp(
        train_embeddings_loader,
        val_embeddings_loader,
        train_embeddings_folder_path,
        val_embeddings_folder_path,
        hyperparameters,
        output_path,
        randomness_seed,
        device,
    )

    # Return the validation EER as the objective metric
    return validation_eer


def objective_svm(
    trial,
    train_embeddings_loader: list,
    val_embeddings_loader: list,
    train_embeddings_folder_path: str,
    val_embeddings_folder_path: str,
    output_path: str,
    randomness_seed: int,
) -> float:
    """
    Objective function for hyperparameter tuning of a Support Vector Machine (SVM) model.

    Parameters
    ----------
    trial : optuna.Trial
        The trial object from Optuna.
    train_embeddings_loader : list
        A list of tuples containing the embeddings and labels for the training data.
    val_embeddings_loader : list
        A list of tuples containing the embeddings and labels for the validation data.
    train_embeddings_folder_path : str
        The path to the folder containing the embeddings for the training data.
    val_embeddings_folder_path : str
        The path to the folder containing the embeddings for the validation data.
    output_path : str
        The path to save the model to.
    randomness_seed : int
        The seed for random number generation.

    Returns
    -------
    float
        The validation EER as the objective metric.
    """
    # Define the hyperparameter search space
    hyperparameters = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "C": trial.suggest_float("C", 0.1, 10.0),
        "kernel": "rbf",
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "randomness_seed": randomness_seed,
    }

    # Call the training function
    validation_eer = train_svm(
        train_embeddings_loader,
        val_embeddings_loader,
        train_embeddings_folder_path,
        val_embeddings_folder_path,
        hyperparameters,
        output_path,
        randomness_seed,
    )

    # Return the validation EER as the objective metric
    return validation_eer


def objective_rf(
    trial,
    train_embeddings_loader: list,
    val_embeddings_loader: list,
    train_embeddings_folder_path: str,
    val_embeddings_folder_path: str,
    output_path: str,
    randomness_seed: int,
) -> float:
    """
    Objective function for hyperparameter tuning of a Random Forest (RF) model.

    Parameters
    ----------
    trial : optuna.Trial
        The trial object from Optuna.
    train_embeddings_loader : list
        A list of tuples containing the embeddings and labels for the training data.
    val_embeddings_loader : list
        A list of tuples containing the embeddings and labels for the validation data.
    train_embeddings_folder_path : str
        The path to the folder containing the embeddings for the training data.
    val_embeddings_folder_path : str
        The path to the folder containing the embeddings for the validation data.
    output_path : str
        The path to save the model to.
    randomness_seed : int
        The seed for random number generation.

    Returns
    -------
    float
        The validation EER as the objective metric.
    """
    # Define the hyperparameter search space
    hyperparameters = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "randomness_seed": randomness_seed,
    }

    # Call the training function
    validation_eer = train_rf(
        train_embeddings_loader,
        val_embeddings_loader,
        train_embeddings_folder_path,
        val_embeddings_folder_path,
        hyperparameters,
        output_path,
        randomness_seed,
    )

    # Return the validation EER as the objective metric
    return validation_eer
