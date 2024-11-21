#!/bin/bash

# AUDIOS - Dataset Preparation Script
# Ensure you have git, git-lfs, unzip, and unrar installed before running.

set -e  # Exit on any error
set -u  # Treat unset variables as errors

echo "Starting dataset preparation..."

# Define base directories
BASE_DIR=$(pwd)/datasets/release
FAKE_VOICES_DIR=$BASE_DIR/fake_voices
REAL_VOICES_DIR=$BASE_DIR/real_voices

# Create dataset folder structure
echo "Creating dataset folders..."
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# Step 1: Download the spoof audios dataset
echo "Downloading the spoof audios dataset..."
git lfs install
git clone https://huggingface.co/datasets/unfake/fake_voices
cd fake_voices/
mv falabrasil-fake-voices/* .
rm -rf .git/ falabrasil-fake-voices/ .gitattributes README.md transcriptions.txt

# Extract files
echo "Extracting compressed files in the fake voices dataset..."
unzip '*.zip'
unrar x '*.rar'
rm *.zip *.rar

# Rename folders to match the pattern
echo "Renaming out-of-pattern folders in fake voices dataset..."
mv MarioJr_Fake/ MarioJr_M014_Fake/
mv PauloSiqueira_Papus_M015_Fake/ PauloSiqueira_M015_Fake/
mv VozesFalsasMilena_F044_Fake/ Milena_F044_Fake/
mv VozesFalsasMyrzaWanderley_F051_Fake/ MyrzaWanderley_F051_Fake/
mv VozesFalsasWalace_M004_Fake/ Walace_M004_Fake/

# Step 2: Download the bona-fide audios dataset
echo "Downloading the bona-fide audios dataset..."
cd "$BASE_DIR"
git clone https://gitlab.com/fb-audio-corpora/alcaim16k-DVD1de4.git
git clone https://gitlab.com/fb-audio-corpora/alcaim16k-DVD2de4.git
git clone https://gitlab.com/fb-audio-corpora/alcaim16k-DVD3de4.git
git clone https://gitlab.com/fb-audio-corpora/alcaim16k-DVD4de4.git

# Delete unnecessary files
echo "Cleaning up repositories..."
find . -name "*.git" -exec rm -rf {} +
find . -name README.md -delete

# Move speaker folders to `real_voices`
echo "Organizing real voices dataset..."
mkdir -p "$REAL_VOICES_DIR"
mv alcaim16k-DVD1de4/* alcaim16k-DVD2de4/* alcaim16k-DVD3de4/* alcaim16k-DVD4de4/* -t "$REAL_VOICES_DIR/"
rm -r alcaim16k-DVD*

# Delete non-WAV files
echo "Deleting non-WAV files..."
find "$REAL_VOICES_DIR" -type f ! -name "*.wav" -delete

# Rename folders to match the pattern
echo "Renaming out-of-pattern folders in real voices dataset..."
cd "$REAL_VOICES_DIR"
mv MarioJr._M014/ MarioJr_M014/
mv PauloSiqueira_Papus_M015/ PauloSiqueira_M015/

# Delete mute audios
echo "Deleting mute audios..."
rm -f Juliana_F028/F028-0914.wav
rm -f Marta_F009/F009-0150.wav

echo "Dataset preparation completed successfully!"
