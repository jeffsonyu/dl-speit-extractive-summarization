import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]
    
    
class DLDataset(Dataset):
    def __init__(self, data_dir, split="train") -> None:
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        
        self.training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
        self.training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in self.training_set])
        self.training_set.remove('IS1002a')
        self.training_set.remove('IS1005d')
        self.training_set.remove('TS3012c')

        self.test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
        self.test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in self.test_set])
        
        self.y_training = []
        with open("data/training_labels.json", "r") as file:
            training_labels = json.load(file)
            
        self.X_training = []
        for transcription_id in self.training_set:
            with open(f"{self.data_dir}/training/{transcription_id}.json", "r") as file:
                transcription = json.load(file)
            
            for utterance in transcription:
                self.X_training.append(utterance["speaker"] + ": " + utterance["text"])
            
            self.y_training += training_labels[transcription_id]
        
        self.X_test = []
        for transcription_id in self.test_set:
            with open(f"{self.data_dir}/test/{transcription_id}.json", "r") as file:
                transcription = json.load(file)
            
            for utterance in transcription:
                self.X_test.append(utterance["speaker"] + ": " + utterance["text"])

    def __len__(self):
        return len(self.X_training) if self.split == "train" else len(self.X_test)

    def __getitem__(self, index):
        item_dict = {}
        if self.split == "train":
            return self.X_training[index], self.y_training[index]
        else:
            return self.X_test[index]
    
if __name__ == "__main__":
    device = "cuda:0"
    data_dir = "data"
    train_dataset = DLDataset(data_dir, split="train")
    print(len(train_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    bert = SentenceTransformer('roberta-base').to(device)
    
    for batch_idx, batch in enumerate(train_dataloader):
        x, y = batch
        y = y.to(device)
        X_encode = bert.encode(x, show_progress_bar=True)
        
        X_encode = torch.tensor(X_encode, dtype=torch.float32, device=device)
        print(X_encode.size())
        break