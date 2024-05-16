# DL Project
Shuyao Qi, Jingyan Wang, Zhenjun Yu

## Overview
This project aims to train and evaluate text classification models using Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures. The models are trained on a dataset of transcriptions, where each transcription consists of multiple utterances. The project leverages the SentenceTransformer library for encoding the text data and applies class weighting to handle imbalanced classes.

## Directory Structure
```
.
├── all-MiniLM-L6-v2
│   └── (model files for MiniLM)
├── ckhpt
│   ├── model_mlp_10.pth
│   └── model_mlp.pth
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

## Requirements

- Python 3.7+
- torch 1.7.1+
- numpy 1.19.2+
- sentence-transformers 0.4.1+
- scikit-learn 0.24.1+


## Installation
1. Clone the repository:
```
git clone https://github.com/jeffsonyu/dl-speit-extractive-summarization.git
cd text-classification
```

2. Create a conda environment and activate it:
```
conda create -n daidaiBird python=3.7
conda activate daidaiBird
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage
To train the model, run the train.py script with the desired parameters. Below is an example command:
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

The testing process is integrated within the training script. After training, the model is evaluated on the test dataset, and the predictions are saved in a JSON file and a submission CSV file.

## Dataset
The dataset should be structured in the following format:

- Training data: `data/train`
- Test data: `data/test`
Each transcription should be stored as a JSON file, where each file contains a list of utterances. Each utterance should be a dictionary with `speaker` and `text` keys.

## Examp;e Transcription Format

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
The trained models are saved in the `ckhpt` directory. The test predictions are saved in `data/test_labels_mlp.json` (for MLP) and `data/test_labels_cnn.json` (for CNN). The submission files are created as `submission_mlp.csv` and `submission_cnn.csv`.

### Submission
To create a submission file, the `make_submission` function is used. It converts the JSON test labels into a CSV file suitable for submission.

## Contact
For any questions or issues, please contact Jeffeson Yu at jeffson-yu@sjtu.edu.cn or Shuyao Qi at nastyapple@sjtu.edu.cn or Jingyan Wang at wangj1ngyan@sjtu.edu.cn.