import os
import numpy as np
import json
import torch
from torch import nn
from dataset import DLDataset

from network.classifier import MLPClassfier, CNNClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report

from make_submission import make_submission

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--device', type=str, default='cuda:0', help='model type')
    parser.add_argument('--data', type=str, default='data', help='data directory')
    parser.add_argument('--type', type=str, default='mlp', help='model path')
    parser.add_argument('--model', type=str, default=None, help='model path')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    return parser.parse_args()

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    train_dataset = DLDataset(args.data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = DLDataset(args.data, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    model_type = args.type
    if model_type == 'mlp':
        model = MLPClassfier(nn_channels=[768, 128, 32, 16, 2])
    elif model_type == 'cnn':
        model = CNNClassifier()
    model.to(args.device)
    
    # Bert encoder
    bert = SentenceTransformer('roberta-base').to(args.device)
    
    if args.model is not None:
        model.load_state_dict(torch.load(os.path.join("ckhpt", args.model)))

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set loss function
    # Prepare weights
    weight_for_class_0 = 1.0
    weight_for_class_1 = 3.0  # add the weight of label 1

    #  Convert to Tensor and move to the correct device
    class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], dtype=torch.float32).to(args.device)

    # # Use weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Train
    os.makedirs('ckhpt', exist_ok=True)
    acc_best = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            y = y.to(args.device)
            
            X_encode = bert.encode(x)
            X_encode = torch.tensor(X_encode).to(args.device)
            
            y_hat = model(X_encode)
            loss = criterion(y_hat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        if (epoch+1) % 5 == 0:
            acc = validate(model, train_loader, bert, args.device)
            torch.save(model.state_dict(), os.path.join('ckhpt', 'model_{}_{:02d}.pth'.format(model_type, epoch+1)))
            if acc > acc_best:
                acc_best = acc
                torch.save(model.state_dict(), os.path.join('ckhpt', 'model_{}.pth'.format(model_type)))
    
    acc = validate(model, train_loader, bert, args.device)
    test(model_type, model, test_dataset, bert, args.device)

def validate(model, loader, bert, device):
    model.eval()
    correct = 0
    total = 0
    
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            X_encode = bert.encode(x)
            X_encode = torch.tensor(X_encode).to(device)

            outputs = model(X_encode)
            _, predicted = torch.max(outputs.data, 1)
            
            preds_all += predicted.cpu().detach().numpy().tolist()
            labels_all += y.cpu().numpy().tolist()
    
    acc = accuracy_score(np.array(labels_all), y_pred=np.array(preds_all))
    print(classification_report(np.array(labels_all), y_pred=np.array(preds_all)))
    print(acc)
    
    return acc

def test(model_type, model, test_dataset, bert, device):
    model.eval()
    
    test_labels = {}
    for transcription_id in test_dataset.test_set:
        with open(f"{test_dataset.data_dir}/test/{transcription_id}.json", "r") as file:
                transcription = json.load(file)
        
        X_test = []
        for utterance in transcription:
            X_test.append(utterance["speaker"] + ": " + utterance["text"])
        
        X_encode = bert.encode(X_test)
        with torch.no_grad():
            X_encode = torch.tensor(X_encode).to(device)

            outputs = model(X_encode)
            _, y_test = torch.max(outputs.data, 1)
        test_labels[transcription_id] = y_test.detach().cpu().numpy().tolist()

    with open(f"data/test_labels_{model_type}.json", "w") as file:
        json.dump(test_labels, file, indent=4)
        
    make_submission(f"data/test_labels_{model_type}.json", f"submission_{model_type}.csv")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)