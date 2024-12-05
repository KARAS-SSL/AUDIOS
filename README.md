# Project Overview: AUDIOS

The project AUDIOS (Audio Deepfake Detection using Self-Supervised Learning) is a machine learning pipeline for audio dataset preprocessing, feature extraction, and model training for the detection of audios generate by AIs.

1. **Dataset Management**: Creating metadata, adding duration and amplitude information, normalizing audio, adding noise, and splitting datasets into training, validation, and test sets.  
2. **Embedding Generation**: Using pre-trained models such as **Wav2Vec2.0** and **HuBERT** to generate audio embeddings for machine learning tasks.  
3. **Visualization**: Tools like UMAP and t-SNE are used to visualize the embeddings in low-dimensional spaces.  
4. **Model Training**: Hyperparameter optimization and training of machine learning models (e.g., MLP, SVM, RF) using embeddings.  

The pipeline automates the steps needed to prepare and train models for downstream tasks, ensuring reproducibility and efficiency.

---

## How to get the Dataset ?

In order know more about the dataset and download it, please follow the instructions in the [Dataset Preparation](DatasetPreparation.md).

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
  fab VisualizeEmbeddingsUMAP <model>
  fab VisualizeEmbeddingsTSNE <model>
  ```

  Replace `<model>` with `wav2vec2` or `hubert`, depending on which you used on the previous step.

  Alternatively, you can visualize the embeddings using both UMAP and t-SNE by running:

  ```bash
  fab VisualizeEmbeddings <model>
  ```

- **Optimize hyperparameters for a specific model:**

  ```bash
  fab OptimizeHyperparameters <backbone> <model>
  ```

  Replace `<backbone>` with `wav2vec2` or `hubert`, depending on the backbone used to generate the embeddings, and `<model>` with `mlp`, `svm`, or `rf`, depending on the model you want to train.  

  Running this command will initiate a search for the best hyperparameters, optimizing the model's performance.

- **(Optional) Train a model:**

  If you want to train a single model, without searching for the best hyperparameters, you can run:

  ```bash
  fab TrainModel <backbone> <model>
  ```

  Replace `<backbone>` with `wav2vec2` or `hubert`, depending on the backbone used to generate the embeddings, and `<model>` with `mlp`, `svm`, or `rf`, depending on the model you want to train.  

  Running this command will train the specified model using predefined hyperparameters.

- **Test a model:**

  ```bash
  fab TestModel <backbone> <model> <run> <study> <gender>
  ```

  Replace:
  
  - `<backbone>` with `wav2vec2` or `hubert`, depending on the backbone used to generate the embeddings
  - `<model>` with `mlp`, `svm`, or `rf`, depending on the model you want to train
  - `<run>` with the number best run (i.e. the run that resulted in the best model)
  - `<study>` with number of the study to which the run belongs
  - `<gender>` with the `M` or `F`, if you want to test the model for a particular gender only (optional).

### 4. **Debugging and Output**

Task output and logs are printed to the console. If an error occurs, review the output for details.

---

### Notes

- Ensure all required files (e.g., dataset metadata) are correctly located in the project directory before running tasks.  
- Tasks are modular, so you can skip or repeat steps as needed.  
- Feel free to extend the `fabfile` to include additional tasks or customizations specific to your workflow.

Created by: Elias Santos Martins, Hayla Pereira Belozi Vasconcellos, Lucas Nogueira Roberto e Matheus Gasparotto Lozano.
