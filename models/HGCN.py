from typing import List

import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from torch.autograd import Variable
import torch.nn.functional as F

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)

class HGCNLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size):
        super(HGCNLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gcn_layers.append(GCNConv(in_size, out_size, normalize=True))
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.num_meta_paths = num_meta_paths

    def forward(self, h, gs):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            if len(g) == 1:
                semantic_embeddings.append(self.gcn_layers[i](h, g[0]).flatten(1))
            else:
                semantic_embeddings.append(self.gcn_layers[i](h, g[0], g[1]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_attention(semantic_embeddings)
    
    
class GATLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(GATLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_channels=in_size, out_channels=out_size, heads=layer_num_heads, dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, node_features, edge_indexes):
        semantic_embeddings = []
        for i, edge_index in enumerate(edge_indexes):
            semantic_embeddings.append(self.gat_layers[i](node_features, edge_index[0]).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)
    
class HGATLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, num_heads, dropout):
        super(HGATLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(GATLayer(
                            num_meta_paths, 
                            hidden_size * num_heads[l - 1],  
                            hidden_size, 
                            num_heads[l], 
                            dropout))

    def forward(self, node_features, edge_indexes):
        for gnn in self.layers:
            node_features = gnn(node_features, edge_indexes)
        return node_features

    
class HGCNForTextClassification(torch.nn.Module):
    def __init__(self, 
                 num_class, 
                 dropout, 
                 embed_dim,
                 edge_mask,
                 conv_type='gcn',
                 word_embedding_bert=False,
                 hidden_size=768, 
                 layers=2, 
                 vocab_size=30522,
                 use_bert=False,
                 finetune_bert=False, 
                 filter_size=[1,2,3,4],
                 profile_feature_size=256,
                 num_filters=192):
        super(HGCNForTextClassification, self).__init__()
        
        self.num_class = num_class
        self.dropout = dropout
        self.embed_dim = embed_dim
        
        self.edge_mask = edge_mask
        self.hgcn_nets = nn.ModuleList()
        self.hidden_size = hidden_size
        self.layers = layers
        self.use_bert = use_bert
        self.finetune_bert = finetune_bert
        self.profile_feature_size = profile_feature_size
        self.word_embedding_bert = word_embedding_bert
        
        self.num_heads = [3] * np.sum(edge_mask)
        
        self.word_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.blstm = torch.nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   bias=True,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)
        if self.use_bert:
            from transformers import AutoTokenizer, AutoModel
            self.bert = AutoModel.from_pretrained("bert-base-chinese")
            pass
        def gen_conv_list(name, in_dim):
            conv_list = []
            for fs in filter_size:
                conv = torch.nn.Conv1d(
                    in_channels=in_dim, out_channels=num_filters, kernel_size=fs)
                conv_list.append(conv)
                self.add_module('%s_conv_%s' % (name, fs), conv)
            return conv_list
        self.sent_conv = gen_conv_list('sentence_conv', self.hidden_size)
        
        self.features_embedding = torch.nn.Parameter(torch.Tensor(self.profile_feature_size, self.hidden_size))
        
        if conv_type == 'gcn':
            for i in range(self.layers):
                self.hgcn_nets.append(
                    HGCNLayer(sum(self.edge_mask), 
                        self.hidden_size, 
                        self.hidden_size))

            self.classifier = nn.Linear(self.hidden_size, self.num_class)
        elif conv_type == 'gat':
            for i in range(self.layers):
                self.hgcn_nets.append(
                    GATLayer(
                        num_meta_paths=sum(self.edge_mask), 
                        in_size=(self.hidden_size if i == 0 else self.hidden_size * 4), 
                        out_size=self.hidden_size,
                        layer_num_heads=4,
                        dropout=dropout,
                    )
                )
            self.classifier = nn.Linear(self.hidden_size * 4, self.num_class)
        elif conv_type == 'hgat':
            for i in range(self.layers):
                self.hgcn_nets.append(
                    HGATLayer(
                        num_meta_paths=sum(self.edge_mask), 
                        in_size=(self.hidden_size if i == 0 else self.hidden_size * 4), 
                        hidden_size=self.hidden_size,
                        num_heads=[4,4,4,4],
                        dropout=dropout,
                    )
                )
            self.classifier = nn.Linear(self.hidden_size * 4, self.num_class)
            
        self.__init_weights__()

    def __init_weights__(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.xavier_uniform_(self.features_embedding)
        torch.nn.init.xavier_uniform_(self.word_embedding.weight)

    def forward(self, input):
        input = input[0]
        node_features, labels = input.x, input.y
        edge_indexes = [
                        [input.w2w_edge_index, input.w2w_edge_attr], 
                        [input.w2sq_edge_index, input.w2sq_edge_attr],
                        [input.sq2w_edge_index, input.sq2w_edge_attr],
                        [input.s2q_edge_index],
                        [input.q2s_edge_index],
                        [input.f2s_edge_index]
                        ]
        node_type = input.node_type
        encode_node_features = []
        for i in range(node_features.shape[0]):
            node_feature = node_features[i]
            if node_type[i].item() == 0:
                if self.use_bert:
                    if self.finetune_bert:
                        text_embedding = self.bert(input_ids=node_feature.unsqueeze(0),
                                                   attention_mask=node_feature.unsqueeze(0)!=0)[1]
                    else:
                        with torch.no_grad():
                            text_embedding = self.bert(input_ids=node_feature.unsqueeze(0),
                                                   attention_mask=node_feature.unsqueeze(0)!=0)[1]
                    encode_feature = text_embedding.squeeze(0)
                else:
                    text_embedding = self.word_embedding(node_feature).unsqueeze(0)
                    lengths: List[int] = [(node_feature!=0).sum()]
                    packed = torch.nn.utils.rnn.pack_padded_sequence(
                        text_embedding, lengths, enforce_sorted=False, batch_first=True
                    )
                    _, [h_n, _] = self.blstm(packed)
                    encode_feature = h_n.mean(dim=0).squeeze(0)
            elif node_type[i].item() == 1:
                if self.word_embedding_bert:
                    pass
                else:
                    encode_feature = torch.matmul(torch.t(node_feature).float(), self.features_embedding)
            else:
                encode_feature = self.word_embedding(node_feature[0]).squeeze(0)
            encode_node_features.append(encode_feature)
        node_features = torch.stack(encode_node_features, dim=0)
        
        edge_indexes = [edge_index for i, edge_index in enumerate(edge_indexes) if self.edge_mask[i] == 1]
            
        for layer in self.hgcn_nets:
            node_features = layer(node_features, edge_indexes)
           
        logits = self.classifier(node_features)[input.label_mask]
        labels = labels[input.label_mask]
        
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_class), labels.view(-1))
        outputs = (loss, logits)
        
        logits = logits.detach().cpu().numpy()[:, 1]
        sorted_logits = np.argsort(-logits)
        tp = sum([1 if labels[i] == 1 else 0 for i in sorted_logits[:6]])
        precision = 0 if tp == 0 else tp / 6
        recall = 0 if tp == 0 else tp / sum(labels).item()
        outputs = outputs + (precision, recall)
        return outputs
