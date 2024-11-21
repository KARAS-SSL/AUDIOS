# AUDIOS - Dataset Preparation

This guide provides instructions for preparing the dataset using the script `prepare_dataset.sh`. The script automates the process for Linux, Mac, and Windows (via Git Bash).

## Prerequisites

Before running the script, make sure you have the following installed on your system:

- **Linux**: [Git](https://git-scm.com/downloads/linux), [Git LFS](https://docs.github.com/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), [`unzip`](https://www.tecmint.com/install-zip-and-unzip-in-linux/), [`unrar`](https://www.geeksforgeeks.org/unrar-files-in-linux/)
- **Windows**: [Git](https://git-scm.com/download/win), [Git LFS](https://docs.github.com/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), [7-Zip](https://www.7-zip.org/) or [WinRAR](https://www.rarlab.com/), `PowerShell`
- **Mac**: [Git](https://git-scm.com/download/mac), [Git LFS](https://docs.github.com/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), [`unzip`](https://formulae.brew.sh/formula/unzip), [The Unarchiver](https://theunarchiver.com/)

## What the Script Does

The script automates the following steps:

1. **Creates the required folder structure**:
   - Creates a `datasets/release/` folder in the root of the repository.
2. **Downloads and organizes the spoof audio dataset**:
   - Clones the dataset from `https://huggingface.co/datasets/unfake/fake_voices`.
   - Extracts `.zip` and `.rar` files and organizes them into folders.
   - Renames folders to follow a consistent naming pattern.
3. **Downloads and organizes the bona-fide audio dataset**:
   - Clones four repositories from `https://gitlab.com/fb-audio-corpora`.
   - Cleans up unnecessary files (e.g., `.git`, `README.md`, non-WAV files).
   - Moves all speaker folders into a single directory and renames as needed.
4. **Deletes specific files**:
   - Removes known mute audio files for better dataset quality.

## How to Use the Script

### 1. Download the Script

If you cloned this repository to your local machine, skip to [step 2](#2-make-the-script-executable-linuxmac).

If you haven't cloned the repository, save the script file [`prepare_dataset.sh`](./prepare_dataset.sh) to your local machine. You can either copy it from the repository or download it directly.

### 2. Make the Script Executable (Linux/Mac)

Run the following command to make the script executable:

```bash
chmod +x prepare_dataset.sh
```

### 3. Run the Script

Navigate to the directory containing the script and execute it:

#### Linux/Mac

```bash
./prepare_dataset.sh
```

#### Windows (Git Bash)

Open **Git Bash**, navigate to the directory containing the script, and run:

```bash
./prepare_dataset.sh
```