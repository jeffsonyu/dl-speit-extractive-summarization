import json
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Any, Tuple
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

from network.gnns import SuperGraphModel, SuperRGCNLayer

def get_file_prefix(is_test: bool) -> List[str]:
    def flatten(list_of_list):
        return [item for sublist in list_of_list for item in sublist]
    path_to_training = Path("data/training")
    path_to_test = Path("data/test")

    training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
    training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
    training_set.remove('IS1002a')
    training_set.remove('IS1005d')
    training_set.remove('TS3012c')

    test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
    test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

    set_to_iter = training_set if not is_test else test_set
    preprefix = path_to_training if not is_test else path_to_test
    
    return list(map(lambda x : preprefix/x, set_to_iter))

# return schema: [file_prefix, index, speaker, content]
def process_text_data(is_test: bool) -> List[Tuple[str, int, str, str]]: 
    ret = []
    for prefix in get_file_prefix(is_test):
        with open(f"{prefix}.json", "r") as file:
            transcription = json.load(file)
            for utterance in transcription:
                ret.append((prefix.name, int(utterance['index']), utterance['speaker'], utterance['text']))
    return ret

    
# return schema: [file_prefix, dst index, src index, relation]
def process_graph_data(is_test: bool) -> List[Tuple[str, int, int, str]]:
    ret = []
    for prefix in get_file_prefix(is_test):
        with open(f"{prefix}.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                processed_line = line.strip().split(' ')
                ret.append((prefix.name, int(processed_line[0]) , int(processed_line[2]) , processed_line[1] ))
    return ret
    
# return schema: [file_prefix, index, label]
def process_label_data(is_test: bool) -> List[Tuple[str, int, int, str]]:
    if is_test:
        assert False, "Test data does not has label file."
    ret = []
    with open("data/training_labels.json", "r") as file:
        training_labels = json.load(file)
    for prefix in get_file_prefix(is_test):
        prefix = prefix.name
        file_label = prefix
        labels = training_labels[file_label]
        for idx,label in enumerate(labels):
            ret.append((prefix, idx, label))
    return ret

def get_statistical_info():
    statistical_info = {'edge_cat': 0, 'node_cat': 0}
    edge_cat = {}
    node_cat = {}
    text_data = process_text_data(False)
    graph_data = process_graph_data(False)
    # label_data = process_label_data(False)

    for edge in graph_data:
        cat = edge[-1]
        if cat not in edge_cat:
            edge_cat[cat] = len(edge_cat)
    
    for node in text_data:
        cat = node[-2]
        if cat not in node_cat:
            node_cat[cat] = len(node_cat)

    statistical_info['edge_cat'] = edge_cat
    statistical_info['node_cat'] = node_cat

    ## print infomation
    print(f"category of node: number: {len(node_cat)} : {[k for k in node_cat.keys()]}")
    print(f"category of edge: number: {len(edge_cat)} : {[k for k in edge_cat.keys()]}")
    return statistical_info


def get_preprocessed_data(is_test: bool):
    get_preprocessed_data_pat = {'data':{}, 'label': {}}
    static_info = get_statistical_info()
    text_data = process_text_data(is_test)
    print(text_data[0])
    graph_data = process_graph_data(is_test)
    print(graph_data[0])
    if not is_test:
        label_data = process_label_data(is_test)
        print(label_data[0])

    bert = SentenceTransformer('roberta-base')
    all_texts = [text[-1] for text in text_data]
    embedded_text = bert.encode(all_texts, show_progress_bar=True)
    print('llm embed shape: ', embedded_text[0].shape)
    for i in range(len(text_data)):
        text_data[i] = (text_data[i][0], text_data[i][1], static_info['node_cat'][text_data[i][2]], embedded_text[i])
    
    for i in range(len(graph_data)):
        graph_data[i] = (graph_data[i][0], graph_data[i][1], graph_data[i][2], static_info['edge_cat'][graph_data[i][3]])
    
    for i in range(len(text_data)):
        if text_data[i][0] not in get_preprocessed_data_pat['data']:
            get_preprocessed_data_pat['data'][text_data[i][0]] = {'nodes': {}, 'edges': {}}
            if not is_test:
                get_preprocessed_data_pat['label'][text_data[i][0]] = {}
        get_preprocessed_data_pat['data'][text_data[i][0]]['nodes'][text_data[i][1]] = (text_data[i][2], text_data[i][3])
    
    for i in range(len(graph_data)):
        src = graph_data[i][2]
        dst = graph_data[i][1]
        get_preprocessed_data_pat['data'][graph_data[i][0]]['edges'][(src, dst)] = graph_data[i][3]

    if not is_test:
        for label in label_data:
            get_preprocessed_data_pat['label'][label[0]][label[1]] = label[2]
    if is_test:
        print('test graphs', get_preprocessed_data_pat['data'].keys())
    else:
        print('train graphs', get_preprocessed_data_pat['data'].keys())
    # k = list(get_preprocessed_data_pat['data'].keys())[0]
    # print(get_preprocessed_data_pat['data'][k]['nodes'].keys())
    # print(get_preprocessed_data_pat['data'][k]['edges'].keys())
    return get_preprocessed_data_pat

def processed_data_to_dgl_graph(preprocessed_data):
    data = preprocessed_data['data']
    graph = {}
    printt = False
    for k in data.keys():
        node_embed_datas = [data[k]['nodes'][i][1] for i in range(len(list(data[k]['nodes'].keys())))]
        node_types_datas = [data[k]['nodes'][i][0] for i in range(len(list(data[k]['nodes'].keys())))]
        edges = list(data[k]['edges'].keys())
        us = torch.tensor([edge[0] for edge in edges])
        vs = torch.tensor([edge[1] for edge in edges])
        edge_types_data = [data[k]['edges'][k2] for k2 in edges]
        if not printt:
            print("Node's Data: ")
            print('shape:', node_embed_datas[0].shape)
            print('type:', node_types_datas[0])
            print("Edge's Data: ")
            print('type:', edge_types_data[0])
            printt = True
        g = dgl.graph((us, vs))
        g.ndata['embd'] = torch.tensor(node_embed_datas, requires_grad=False)
        g.ndata['type'] = torch.tensor(node_types_datas, requires_grad=False)
        g.edata['type'] = torch.tensor(edge_types_data, requires_grad=False)
        graph[k] = g
    return graph


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}") 



def train_pipe(preprocessed_data, labeler, model: nn.Module):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    epoch_num = 20
    losser = torch.nn.BCELoss()
    model.cuda()
    for epoch in range(1, epoch_num + 1):
        print(f'#### epoch {epoch} ####')
        count_parameters(model)
        cnt = 0
        data = processed_data_to_dgl_graph(preprocessed_data)
        gl  = len(data.keys())
        for graph_name in data.keys():
            cnt += 1
            g = data[graph_name]
            g = g.to("cuda")
            g.ndata["embd"] = g.ndata["embd"].cuda()
            g.ndata["type"] = g.ndata["type"].cuda()
            g.edata["type"] = g.edata["type"].cuda()
            preds = model(g).reshape(-1)

            labels = torch.tensor([labeler[graph_name][i] for i in range(len(labeler[graph_name]))], dtype=torch.float32, device='cuda')
            loss = losser(preds, labels)
            loss.backward()
            optimizer.step()
            print(f"{cnt}/{gl}, loss: {loss.item()}")
            # print(torch.sum(preds>0.5 == labels>0.5)/labels.shape[0])
            
    print(classification_report(labels.cpu().numpy(), y_pred=(preds>0.5).cpu().detach().numpy()))
            # print(preds[0])
            # print(labels[0])
                  

if __name__ == '__main__':
    preprocessed_data = get_preprocessed_data(False)
    model = SuperGraphModel(input_dim=768, 
                            hidden_dims=[64, 32], 
                            out_dim=1, 
                            num_relation_types=16, 
                            num_node_types=4, 
                            num_bases=1)
    train_pipe(preprocessed_data, preprocessed_data['label'], model)

