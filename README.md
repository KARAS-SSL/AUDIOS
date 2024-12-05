# Project Overview: AUDIOS

The project AUDIOS (Audio Deepfake Detection using Self-Supervised Learning) is a machine learning pipeline for audio dataset preprocessing, feature extraction, and model training for the detection of audios generate by AIs.

1. **Dataset Management**: Creating metadata, adding duration and amplitude information, normalizing audio, adding noise, and splitting datasets into training, validation, and test sets.  
2. **Embedding Generation**: Using pre-trained models such as **Wav2Vec2.0** and **HuBERT** to generate audio embeddings for machine learning tasks.  
3. **Visualization**: Tools like UMAP and t-SNE are used to visualize the embeddings in low-dimensional spaces.  
4. **Model Training**: Hyperparameter optimization and training of machine learning models (e.g., MLP, SVM, RF) using embeddings.  

The pipeline automates the steps needed to prepare and train models for downstream tasks, ensuring reproducibility and efficiency.

---

## How to get the Dataset ?

In order know more about the dataset and download it, please follow the instructions in the [Dataset Preparation](DatasetPreparation.md). To train or test the models on the dataset, some additional steps are required, including processing the audio files, generating CSV files, and splitting the data. Detailed commands and instructions for these steps are outlined in the next section.

---

## How to run the Project ?

The `fabfile.py` contains tasks that automate the pipeline using **Fabric**, a Python library for task automation. Below is a step-by-step guide to using it within this project:

### 1. **Setup**

First things first, make sure you have all dependencies installed with the following command:

```bash
pip install -r requirements.txt
```

### 2. **Available Tasks**

You can list all available tasks using:

```bash
fab -l
```

If you have any doubt on how to run a certain task (i.e., the arguments it accpets), run:

```bash
fab -h TASKNAME
```

### 3. **Running Specific Tasks**

Run individual tasks using the command `fab <task_name>`. For example, run this sequence of tasks to follow our pipeline:

- **Generate dataset metadata:**

  ```bash
  fab GenerateDatasetCSV
  ```

- **Add duration and amplitude information:**

  ```bash
  fab AddDatasetDuration
  fab AddDatasetAmplitude
  ```

- **Normalize dataset amplitudes:**

  ```bash
  fab NormalizeDataset
  ```

- **Split the dataset into pretext and downstream tasks:**

  ```bash
  fab SplitDataset
  ```

- **Generate embeddings with Wav2Vec2.0 or HuBERT:**

  ```bash
  fab GenerateEmbeddingsWav2vec2
  # or
  fab GenerateEmbeddingsHubert
  ```

- **Visualize embeddings using UMAP or t-SNE:**

  ```bash
  fab VisualizeEmbeddingsUMAP --backbone=<backbone>
  fab VisualizeEmbeddingsTSNE --backbone=<backbone>
  ```

  Replace `<backbone>` with `wav2vec2` or `hubert`, depending on which you used on the previous step.

  Alternatively, you can visualize the embeddings using both UMAP and t-SNE by running:

  ```bash
  fab VisualizeEmbeddings --backbone=<backbone>
  ```

- **Optimize hyperparameters for a specific model:**

  ```bash
  fab OptimizeHyperparameters --backbone=<backbone> --classifier=<classifier>
  ```

  Replace `<backbone>` with `wav2vec2` or `hubert`, depending on the backbone used to generate the embeddings, and `<classifier>` with `mlp`, `svm`, or `rf`, depending on the model you want to train.  

  Running this command will initiate a search for the best hyperparameters, optimizing the model's performance.

- **(Optional) Train a model:**

  If you want to train a single model, without searching for the best hyperparameters, you can run:

  ```bash
  fab TrainModel --backbone=<backbone> --classifier=<classifier>
  ```

  Replace `<backbone>` with `wav2vec2` or `hubert`, depending on the backbone used to generate the embeddings, and `<classifier>` with `mlp`, `svm`, or `rf`, depending on the model you want to train.  

  Running this command will train the specified model using predefined hyperparameters.

- **Test a model:**

  ```bash
  fab TestModel --backbone=<backbone> --classifier=<classifier> --run-number=<run> --study-number=<study> --gender=<gender>
  ```

  Replace:
  
  - `<backbone>` with `wav2vec2` or `hubert`, depending on the backbone used to generate the embeddings
  - `<classifier>` with `mlp`, `svm`, or `rf`, depending on the model you want to train
  - `<run>` with the number best run (i.e. the run that resulted in the best model)
  - `<study>` with number of the study to which the run belongs
  - `<gender>` with the `M` or `F`, if you want to test the model for a particular gender only (optional).

### 4. **Debugging and Output**

Task output and logs are printed to the console. If an error occurs, review the output for details.

---

## Directory Structure

The `src` folder is organized as follows:

```plaintext
src/
├── models/
│   ├── mlp/
│   │   ├── mlp_model.py         # Core implementation of the MLP model
│   │   ├── test_mlp.py          # Scripts for testing MLP performance
│   │   ├── train_mlp.py         # Script for training the MLP model
│   ├── random_florest/
│   │   ├── test_rf.py           # Script for testing the Random Forest model
│   │   ├── train_rf.py          # Script for training the Random Forest model
│   ├── svm/
│       ├── test_svm.py          # Script for testing the SVM model
│       ├── train_svm.py         # Script for training the SVM model
├── notebooks/
│   ├── dataset_exploration.ipynb # Notebook for exploring the dataset
│   ├── dataset_normalization.ipynb # Notebook for applying normalization
│   ├── dataset_split.ipynb      # Notebook for dataset splitting
├── utils/
│   ├── dataset.py               # Utility functions for dataset handling
│   ├── eer.py                   # Utility functions for Equal Error Rate (EER) computation
│   ├── embeddings.py            # Utilities for generating the Backbones embeddings
├── visualize/
│   ├── tsne.py                  # Script for t-SNE visualizations
│   ├── umap.py                  # Script for UMAP visualizations
├── .gitignore                   # Git ignore file for excluding unnecessary files
├── DatasetPreparation.md        # Documentation for dataset preparation
├── fabfile.py                   # Automation script for common tasks
├── LICENSE                      # License file
```

## Key Components

### `models/`
This folder contains implementations of machine learning models, split into subdirectories for each model type:
- **`mlp/`**: Implements a Multi-Layer Perceptron (MLP) model.
- **`random_florest/`**: Contains Random Forest-related scripts.
- **`svm/`**: Contains support vector machine (SVM) scripts.

### `notebooks/`
Contains Jupyter Notebooks for exploratory and preparatory tasks like dataset exploration, normalization, and splitting. These mostly uses the function in models and utils folders.

Also contains the main Notebook for the [YAMNet Baseline Model](src/notebooks/CNN_baseline_model.ipynb) training and testing.

### `utils/`
Includes utility scripts for:
- Handling datasets.
- Computing metrics like EER.
- Working with embeddings.

### `visualize/`
Contains scripts for visualizing embeddings using t-SNE and UMAP techniques.

### `fabfile.py`
A Fabric script for automating tasks like running training pipelines or cleaning directories, as detailed in the previous section.

### `DatasetPreparation.md`
Documentation for preparing the dataset, explaining preprocessing and splitting steps.

---

## Runs folder

The `runs` folder stores the outputs generated by the training and testing pipelines. If it does not already exist, it will be created automatically when running the tasks. This folder contains subdirectories, each corresponding to the training of a specific model or a batch of models in the case of hyperparameter searches. Each subdirectory may include the following:

- **Model weights**: Saved in appropriate files.
- **Loss data**: `.npy` files containing training and validation losses.
- **Hyperparameters**: Stored in JSON files.
- **Testing metrics**: A JSON file summarizing the evaluation results.
- **Visualizations**: Plots such as the confusion matrix and other relevant charts.

This structure helps organize and track the results for different experiments efficiently.

___

Let me know if you'd like any further updates or adjustments!
---

### Notes

- Ensure all required files (e.g., dataset metadata) are correctly located in the project directory before running tasks.  
- Tasks are modular, so you can skip or repeat steps as needed.  
- Feel free to extend the `fabfile` to include additional tasks or customizations specific to your workflow.

---

### Credits

Created by: Elias Santos Martins, Hayla Pereira Belozi Vasconcellos, Lucas Nogueira Roberto e Matheus Gasparotto Lozano.
