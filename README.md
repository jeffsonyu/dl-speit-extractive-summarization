# DL Project
Shuyao Qi, Jingyan Wang, Zhenjun Yu

## Overview
This project aims to train and evaluate text classification models using different approaches: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), classical machine learning models, and Graph Neural Networks (GNNs). The models are trained on a dataset of transcriptions, where each transcription consists of multiple utterances. The project leverages the SentenceTransformer library for encoding the text data and applies class weighting to handle imbalanced classes.

## Directory Structure
```
.
├── ckhpt
│   ├── model_mlp.pth
│   └── ...
├── data
│   ├── test
│   │   └── (test data files)
│   ├── training
│   │   └── (training data files)
│   ├── test_labels_naive_baseline.json
│   ├── test_labels_text_baseline.json
│   └── training_labels.json
├── network
│   ├── classifier.py
│   ├── gnns_new.py
│   └── gnns.py
├── .gitignore
├── baseline.py
├── dataset.py
├── make_submission.py
├── model_rgcn.ipynb
├── README.md
├── requirements.txt
├── submission_naive_baseline.csv
├── test_labels_naive_baseline.json
├── test.py
├── train_graph.py
├── train_naive.py
└── train.py

```


## Installation
1. Clone the repository:
```
git clone https://github.com/jeffsonyu/dl-speit-extractive-summarization.git
cd dl-speit-extractive-summarization
```

2. Create a conda environment and activate it:
```
conda create -n daidaiBird python=3.9
conda activate daidaiBird
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage
### Training Neural Network Models (MLP/CNN)

To train the model, run the `train.py` script with the desired parameters. Below is an example command:
```
python train.py --device cuda:0 --data data --type mlp --batch-size 64 --epochs 20 --lr 0.001 --seed 42
```

- `--device`: The device to use for training (e.g., cuda:0, cpu).
- `--data`: The directory where the data is stored.
- `--type`: The type of model to train (mlp or cnn).
- `--model`: Path to a pre-trained model to continue training (optional).
- `--batch-size`: The batch size for training.
- `--epochs`: The number of epochs to train.
- `--lr`: The learning rate.
- `--seed`: The random seed for reproducibility.

### Training Baseline Models (XGBoost/Random Forest)
To train the baseline models, run the `train_naive.py` script:
```
python train_naive.py
```
This script uses BERT embeddings and trains XGBoost and Random Forest classifiers. It also evaluates the models and generates submission files.

### Training Graph-Based Models (GNN)
To train the graph-based models, run the `train_graph.py` script:
```
python train_graph.py
```
This script processes the data into graph structures, trains a Graph Neural Network model, and evaluates its performance.

## Dataset
The dataset should be structured in the following format:

- Training data: `data/train`
- Test data: `data/test`
Each transcription should be stored as a JSON file, where each file contains a list of utterances. Each utterance should be a dictionary with `speaker` and `text` keys.

## Example Transcription Format

```
[
    {
        "speaker": "Speaker1",
        "text": "Hello, how are you?"
    },
    {
        "speaker": "Speaker2",
        "text": "I'm fine, thank you!"
    }
]

```

## Model Architectures
### MLP Classifier
The MLP classifier is defined with a series of fully connected layers specified in the `MLPClassifier` class in `network/classifier.py`.

### CNN Classifier
The CNN classifier architecture is defined in the `CNNClassifier` class in `network/classifier.py`.

### Output
The trained models are saved in the ckhpt directory. The test predictions are saved in JSON files, and submission files are generated as follows:

- `train.py`: Generates `submission_mlp.csv` or `submission_cnn.csv`.
- `train_naive.py`: Generates `submission_baseline.csv` and `submission_baseline_rf.csv`.
- `train_graph.py`: Generates `submission_rgcn.csv`.

### Submission
To create a submission file, the `make_submission` function is used. It converts the JSON test labels into a CSV file suitable for submission.

## Contact
For any questions or issues, please contact Jeffeson Yu at jeffson-yu@sjtu.edu.cn or Shuyao Qi at nastyapple@sjtu.edu.cn or Jingyan Wang at wangj1ngyan@sjtu.edu.cn.