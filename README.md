# Hope Classification Project

A machine learning project that uses BERT-based models to classify text into hope-related categories, developed for Texas Tech University's HPCC (High Performance Computing Center).

## Overview

This project implements two classification tasks:

1. Binary Classification: Categorizing text as either "Hope" or "Not Hope"
2. Multiclass Classification: Categorizing text into five categories:
   - Not Hope
   - Generalized Hope
   - Realistic Hope
   - Unrealistic Hope
   - Sarcasm

## Technical Stack

- Python 3.8+
- TensorFlow
- Hugging Face Transformers (BERT)
- pandas
- scikit-learn

## Project Structure

```
/Hope/
├── hope_classifier.py      # Main classification script
├── launch_hope.sh         # SLURM job submission script
├── en_train.csv           # Training dataset
├── en_dev.csv            # Development/test dataset
├── out/                  # Output directory
│   ├── hope_output.log   # Training logs
│   └── hope_results.txt  # Classification results
└── bert_tokenizer/       # Saved BERT tokenizer
```

## Features

- Text preprocessing and cleaning
- BERT tokenization
- Binary hope classification
- Multiclass hope classification
- Model checkpointing
- Performance evaluation
- Result logging

## Model Architecture

- Base Model: BERT (bert-base-uncased)
- Binary Classification: 2 output classes
- Multiclass Classification: 5 output classes
- Training Parameters:
  - Learning Rate: 2e-5
  - Optimizer: Adam
  - Loss: SparseCategoricalCrossentropy
  - Batch Size: 16
  - Epochs: 3

## Usage

1. Ensure you have access to TTU's HPCC (Quanah)
2. Set up the conda environment:

```bash
conda create -n hopeenv python=3.8
conda activate hopeenv
pip install tensorflow transformers pandas scikit-learn
```

3. Submit the job:

```bash
sbatch launch_hope.sh
```

## Resource Requirements

- Node: 1
- Tasks per node: 1
- Memory: 40GB per CPU
- Runtime: 14 hours
- Partition: nocona

## Output

The models will save:

- Binary classifier: `/lustre/work/cijezue/Hope/bert_binary_model`
- Multiclass classifier: `/lustre/work/cijezue/Hope/bert_multi_model`
- Results: `/lustre/work/cijezue/Hope/out/hope_results.txt`

## Authors

### Lead Developer

- **Name**: Ebuka Ijezue
- **Email**: cijezue@ttu.edu
- **Institution**: Texas Tech University

### Collaborators

- **Name**: Fredrick Eneye Tania-Amanda
- **Email**: tafredri@ttu.edu
- **Institution**: Texas Tech University

## License

MIT License

## Acknowledgments

- Texas Tech University HPCC
- Hugging Face Transformers
- BERT developers
