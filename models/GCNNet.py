import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SplineConv
import numpy as np

class GCNNet(torch.nn.Module):
    def __init__(self, 
                 node_size, 
                 embed_dim,
                 embedding_finetune = False,
                 hidden_dim = 512, 
                 num_class = 2,
                 dropout = 0.5,
                 layers = 2):
        super(GCNNet, self).__init__()
        self.node_size = node_size
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(num_embeddings=node_size, embedding_dim=embed_dim)
        self.embedding_finetune = embedding_finetune
        self.convs = torch.nn.ModuleList([GCNConv(embed_dim, hidden_dim, normalize=False)])
        self.convs.extend([GCNConv(hidden_dim, hidden_dim, normalize=False) for i in range(layers - 2)])
        self.convs.append(GCNConv(hidden_dim, num_class, normalize=False))
        self._weight_init_()
        self.dropout = dropout

    def _weight_init_(self):
        for conv in self.convs:
            init_range = np.sqrt(6.0/(conv.weight.shape[0] + conv.weight.shape[1]))
            torch.nn.init.uniform_(conv.weight, -init_range, init_range)
            
        if not self.embedding_finetune:
            self.embed.weight.requires_grad = False 
            self.embed.weight.copy_(torch.tensor(torch.eye(self.node_size, self.embed_dim), dtype=torch.float32))

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = edge_index.squeeze(), edge_attr.squeeze()
        x = self.embed(x).squeeze()
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, edge_index, edge_weight=edge_attr))
        return F.log_softmax(x, dim=1)
