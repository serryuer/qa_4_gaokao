import os, sys, json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SplineConv
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel

class BertMatch(torch.nn.Module):
    def __init__(self):
        super(BertMatch, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=768, num_layers=2, bias=True, batch_first=True, dropout=0.3, bidirectional=True)
        self.sentence_linear = torch.nn.Linear(768 * 2, 768)
        self.classifier = torch.nn.Linear(768 * 2, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, inputs):
        input_ids, input_mask, segment_ids, labels = inputs
        embeddings = self.bert(input_ids=input_ids.squeeze(0), attention_mask=input_mask.squeeze(0), token_type_ids=segment_ids.squeeze(0))[1].squeeze(0)     
        question_embedding = embeddings[0].squeeze()
        sentences_embedding = embeddings[1:]
        sentences_embedding = self.lstm(sentences_embedding.unsqueeze(0))[0].squeeze()
        sentences_embedding = self.sentence_linear(sentences_embedding)
        question_embedding = question_embedding.unsqueeze(0).expand(sentences_embedding.shape)
        embeddings = torch.cat([question_embedding, sentences_embedding], dim=1)
        logits = self.classifier(embeddings)
        output_dict = {'logits': logits}
        output_dict['loss'] = self.loss_fn(logits, labels.squeeze(0).long())
        return output_dict['loss'], output_dict['logits']
        
        
        


