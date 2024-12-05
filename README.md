
# Project Overview: AUDIOS
The project AUDIOS (Audio Deepfake Detection using Self-Supervised Learning) is a machine learning pipeline for audio dataset preprocessing, feature extraction, and model training for the detection of audios generate by AIs.

1. **Dataset Management**: Creating metadata, adding duration and amplitude information, normalizing audio, adding noise, and splitting datasets into training, validation, and test sets.  
2. **Embedding Generation**: Using pre-trained models such as **Wav2vec2** and **HuBERT** to generate audio embeddings for machine learning tasks.  
3. **Visualization**: Tools like UMAP and t-SNE are used to visualize the embeddings in low-dimensional spaces.  
4. **Model Training**: Hyperparameter optimization and training of machine learning models (e.g., MLP, SVM, RF) using embeddings.  

The pipeline automates the steps needed to prepare and train models for downstream tasks, ensuring reproducibility and efficiency.

---

## How to Run the Fabfile

The `fabfile` contains tasks that automate the pipeline using **Fabric**, a Python library for task automation. Below is a step-by-step guide to using it:

### 1. **Setup**
Ensure you have Python installed along with Fabric:
```bash
pip install fabric
```

### 2. **Available Tasks**
You can list all available tasks using:
```bash
fab -l
```

### 3. **Running Specific Tasks**
Run individual tasks using the command `fab <task_name>`. For example:

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

- **Generate embeddings with Wav2vec2 or HuBERT:**
  ```bash
  fab GenerateEmbeddingsWav2vec2
  fab GenerateEmbeddingsHubert
  ```

- **Visualize embeddings using UMAP or t-SNE:**
  ```bash
  fab VisualizeEmbeddingsUMAP
  fab VisualizeEmbeddingsTSNE
  ```

- **Optimize hyperparameters for a specific model:**
  Replace `<model>` with `mlp`, `svm`, or `rf`:
  ```bash
  fab OptimizeHyperparameters:<model>
  ```

- **Train a model:**
  ```bash
  fab TrainModel
  ```

### 4. **Customizing Tasks**
Many tasks accept arguments for flexibility. For example, to normalize the dataset into a specific directory:
```bash
fab NormalizeDataset --use-same-dir=True
```

### 5. **Debugging and Output**
Task output and logs are printed to the console. If an error occurs, review the output for details.

---

### Notes
- Ensure all required files (e.g., dataset metadata) are correctly located in the project directory before running tasks.  
- Tasks are modular, so you can skip or repeat steps as needed.  
- Feel free to extend the `fabfile` to include additional tasks or customizations specific to your workflow.

Created by: Elias Santos Martins, Hayla Pereira Belozi Vasconcellos, Lucas Nogueira Roberto e Matheus Gasparotto Lozano.




