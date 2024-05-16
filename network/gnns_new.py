import json
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

class SuperRGCNLayer(nn.Module):
    def __init__(
        self,
        in_feature_dim: int,
        out_feature_dim: int,
        num_bases: int,
        num_relation_types: int,
        num_node_types: int,
        add_bias=None,
        activation=None,
    ):
        super(SuperRGCNLayer, self).__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.num_base = num_bases
        self.num_node_types = num_node_types
        self.num_relation_types = num_relation_types
        self.bias = None
        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.Tensor(1,out_feature_dim))
        self.activation = activation
        self.weight = nn.Parameter(
            torch.Tensor(num_bases, in_feature_dim, out_feature_dim)
        )

        ## change here
        self.w_comp = nn.Parameter(
            torch.Tensor(num_relation_types * num_node_types * num_node_types, num_bases)
        )

        self.node_type_embd = nn.Parameter(
            torch.Tensor(num_node_types, in_feature_dim)
        )

        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.node_type_embd
        )
        if num_bases < num_relation_types:
            nn.init.xavier_uniform_(
                self.w_comp, gain=nn.init.calculate_gain("relu")
            )
        if not self.bias is None:
            nn.init.xavier_uniform_(
                self.bias, gain=nn.init.calculate_gain("relu")
            )

        
    def forward(self, g: dgl.DGLGraph):
        weight = self.weight.view(
            self.out_feature_dim, self.num_base, self.in_feature_dim
        )
        # print(torch.matmul(self.w_comp, weight).shape, self.w_comp.shape, weight.shape, self.num_relation_types, self.out_feature_dim, self.in_feature_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_relation_types * self.num_relation_types, self.out_feature_dim, self.in_feature_dim)
        ## change here
        def message_func(edges):
            # relation_type = edges.data['type']
            # node_type = edges.src['type'] 
            # node_val = edges.src['embd'] + self.node_type_embd[node_type]
            # relation_transition_mtx = weight[relation_type]
            # shape = node_val.shape
            # output_shape0 = relation_transition_mtx.shape[1]
            # ret = F.dropout(torch.bmm(relation_transition_mtx, node_val.view(shape[0], shape[1], 1)).view(shape[0], output_shape0), p=0.1)
            relation_type = edges.data['type']
            src_node_type = edges.src['type'] 
            dst_node_type = edges.dst['type']
            relation_type = src_node_type * self.num_node_types * self.num_relation_types + dst_node_type * self.num_relation_types + relation_type
            node_val = edges.src['embd'] # + self.node_type_embd[node_type]
            relation_transition_mtx = weight[relation_type]
            shape = node_val.shape
            output_shape0 = relation_transition_mtx.shape[1]
            ret = F.dropout(torch.bmm(relation_transition_mtx, node_val.view(shape[0], shape[1], 1)).view(shape[0], output_shape0), p=0.1)
            return {'msg': ret}
        
        def apply_func(nodes):
            embd = nodes.data['embd']
            types = nodes.data['type']
            if self.add_bias:
                embd += self.bias
            if self.activation:
                embd = self.activation(embd)
            return {"embd": embd, "type": types}
        g.update_all(message_func, fn.sum(msg='msg', out='embd'), apply_func)

class SuperGraphModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        out_dim,
        num_relation_types,
        num_node_types,
        num_bases=-1
    ):
        super(SuperGraphModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.num_relation_types = num_relation_types
        self.num_node_types = num_node_types
        self.num_bases = num_bases

        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        for _ in range(1,len(self.hidden_dims)):
            h2h = self.build_hidden_layer(_)
            self.layers.append(h2h)
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        return SuperRGCNLayer(
            self.input_dim,
            self.hidden_dims[0],
            self.num_bases,
            self.num_relation_types,
            self.num_node_types,
            add_bias=True,
            activation=F.relu
        )

    def build_hidden_layer(self,index):
        return SuperRGCNLayer(
            self.hidden_dims[index-1],
            self.hidden_dims[index],
            self.num_bases,
            self.num_relation_types,
            self.num_node_types,
            add_bias=False,
            activation=F.relu
        )

    def build_output_layer(self):
        return SuperRGCNLayer(
            self.hidden_dims[-1],
            self.out_dim,
            self.num_bases,
            self.num_relation_types,
            self.num_node_types,
            add_bias=False,
            activation=F.sigmoid,
        )

    def forward(self, g):
        for layer in self.layers:
            layer(g)
        return g.ndata["embd"]