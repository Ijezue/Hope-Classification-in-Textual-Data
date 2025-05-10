# Hope Classification Project

A machine learning project that compares transformer-based models (BERT, GPT-2, and DeBERTa) for hope classification in textual data, developed for Texas Tech University.

## Overview

This project implements two classification tasks across two distinct implementations:

1. **Original BERT Implementation**:

   - Binary classification: "Hope" vs. "Not Hope"
   - Multiclass classification: Five hope-related categories

2. **Extended Implementation** comparing three transformer architectures:
   - BERT (bidirectional encoder)
   - GPT-2 (unidirectional autoregressive)
   - DeBERTa (disentangled attention)

The project evaluates model performance, computational efficiency, and classification patterns to determine the optimal architecture for hope detection.

## Classification Tasks

- **Binary Classification**: Categorizing text as either "Hope" or "Not Hope"
- **Multiclass Classification**: Categorizing text into five categories:
  - Not Hope
  - Generalized Hope
  - Realistic Hope
  - Unrealistic Hope
  - Sarcasm

## Technical Stack

- Python 3.8+
- TensorFlow
- Hugging Face Transformers (BERT, GPT-2, DeBERTa)
- pandas
- scikit-learn
- matplotlib & seaborn (for visualization)
- Jupyter Notebooks (for implementation and analysis)

## Project Structure

```
/
├── bert-classifier.ipynb                # Original BERT implementation notebook
├── bert.py                              # Python script of original BERT implementation
├── Extended_Implementation_Hope_vs_NotHope.ipynb  # Extended implementation comparing all models
├── launch_hope.sh                       # SLURM job submission script for HPCC
├── submit.py                            # Prediction script for test data
├── en_train.csv                         # Training dataset
├── en_dev.csv                           # Development/test dataset
├── extended-implementation-results/     # Results from extended model comparison
│   ├── accuracy_comparison.png          # Accuracy comparison visualization
│   ├── bert_binary_confusion_matrix.png # BERT binary classification confusion matrix
│   ├── bert_multiclass_confusion_matrix.png
│   ├── comparison_results.txt           # Detailed performance metrics
│   ├── deberta_binary_confusion_matrix.png
│   ├── deberta_multiclass_confusion_matrix.png
│   ├── gpt2_binary_confusion_matrix.png
│   ├── gpt2_multiclass_confusion_matrix.png
│   ├── model_comparison_metrics.csv     # Tabulated performance comparison
│   └── training_time_comparison.png     # Computational efficiency comparison
└── README.md                            # Project documentation
```

## Model Implementations

### Original BERT Implementation

- **Model**: BERT (bert-base-uncased)
- **Features**: Minimal text preprocessing, direct tokenization
- **Environment**: HPCC with NVIDIA A100 GPUs
- **Training Parameters**:
  - Batch Size: 8
  - Learning Rate: 2e-5
  - Epochs: 3
  - Max Sequence Length: 128

### Extended Implementation Comparison

- **Models**: BERT, GPT-2, DeBERTa (base versions)
- **Features**: Basic text preprocessing (lowercase, URL removal, hashtag removal, punctuation removal)
- **Environment**: Google Colab with NVIDIA T4 GPUs
- **Training Parameters**:
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Epochs: 3
  - Max Sequence Length: 128

## Key Findings

- BERT achieved the highest performance for both binary (84.49%) and multiclass (72.03%) classification in the extended study
- Original BERT implementation achieved 83.65% for binary and 74.87% for multiclass classification
- GPT-2 showed remarkable strength in sarcasm detection (92.46% recall)
- DeBERTa required nearly double BERT's training time while delivering comparable or slightly lower performance
- Implementation details (preprocessing, batch size, computational environment) significantly impact model performance

## Usage

### Local Development

1. Clone the repository
2. Install dependencies:

```bash
pip install tensorflow transformers pandas scikit-learn jupyter matplotlib seaborn
```

3. Run the Jupyter notebooks:
   - `bert-classifier.ipynb` for the original BERT implementation
   - `Extended_Implementation_Hope_vs_NotHope.ipynb` for the extended comparison

### HPCC Deployment (TTU Quanah)

1. Set up the conda environment:

```bash
conda create -n hopeenv python=3.8
conda activate hopeenv
pip install tensorflow transformers pandas scikit-learn
```

2. Submit the job:

```bash
sbatch launch_hope.sh
```

## Resource Requirements

- **Original Implementation (HPCC)**:

  - Node: 1
  - Memory: 40GB per CPU
  - GPU: NVIDIA A100
  - Runtime: ~14 hours

- **Extended Implementation (Google Colab)**:
  - GPU: NVIDIA T4
  - Runtime: Varies by model (443s - 948s per task)

## Authors

- **Ebuka Ijezue** - _Lead Developer_ - [cijezue@ttu.edu]
- **Fredrick Eneye Tania-Amanda** - _Collaborator_ - [tafredri@ttu.edu]

## License

MIT License

## Acknowledgments

- Texas Tech University HPCC
- Hugging Face Transformers
- PolyHope shared task at IberLEF 2025
