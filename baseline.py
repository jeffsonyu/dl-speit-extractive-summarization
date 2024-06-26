import json
from pathlib import Path

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

path_to_training = Path("data/training")
path_to_test = Path("data/test")

#####
# training and test sets of transcription ids
#####
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

#####
# naive_baseline: all utterances are predicted important (label 1)
#####
test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    test_labels[transcription_id] = [1] * len(transcription)

with open("data/test_labels_naive_baseline.json", "w") as file:
    json.dump(test_labels, file, indent=4)

#####
# text_baseline: utterances are embedded with SentenceTransformer, then train a classifier.
#####
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
bert = SentenceTransformer('roberta-base')

y_training = []
with open("data/training_labels.json", "r") as file:
    training_labels = json.load(file)
X_training = []
for transcription_id in training_set:
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    for utterance in transcription:
        X_training.append(utterance["speaker"] + ": " + utterance["text"])
    
    y_training += training_labels[transcription_id]

X_training = bert.encode(X_training, show_progress_bar=True)

print(X_training.shape)
scale_pos_weight = (len(y_training)/sum(y_training)) ** 0.9
clf = XGBClassifier(n_estimators=1024, max_depth=3, learning_rate=0.1, objective='binary:logistic', device='cuda', scale_pos_weight=scale_pos_weight)
clf.fit(X_training, y_training)

from sklearn.metrics import accuracy_score, classification_report

y_training_pred = clf.predict(X_training)
print(classification_report(y_true= y_training, y_pred=y_training_pred))
print(accuracy_score(y_training, y_training_pred))

test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    X_test = []
    for utterance in transcription:
        X_test.append(utterance["speaker"] + ": " + utterance["text"])
    
    X_test = bert.encode(X_test)

    y_test = clf.predict(X_test)
    test_labels[transcription_id] = y_test.tolist()

with open("data/test_labels_text_baseline.json", "w") as file:
    json.dump(test_labels, file, indent=4)
