import logging
# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)
import random, os, sys, json
import torch
import argparse
from sklearn.metrics import *
from transformers import AdamW, BertTokenizer
from torch.utils.data import DataLoader
from torch_geometric.data import DataListLoader

from datasets.QAGraphDataset import QAGraphDataset
from datasets.QATextDataset import QATextDataset
from trainer import Train
from models.BertMatch import BertMatch
from models.HGCN import HGCNForTextClassification


parser = argparse.ArgumentParser(description="text gcn with pytorch + torch_geometric")
parser.add_argument('-lr', type=float, default=1e-5, help='initial learning rate [default: 0.001]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 20]')
parser.add_argument('-gcn_layers',type=int,default=2,help='the network layer [default 2]')
parser.add_argument('-hidden_size',type=int,default=768,help='number of hidden size for one rnn cell [default: 768]')
parser.add_argument('-random_seed',type=int,default=1024,help='attention size')
parser.add_argument('-device', type=int, default=1, help='device to use for iterate data, -1 mean cpu,1 mean gpu [default: -1]')
parser.add_argument('-model_name', type=str, default='qa-graph-hgat', help='model name')
parser.add_argument('-early_stop_patience', type=int, default=3, help='early stop patience')
parser.add_argument('-loss_weight',type=int,default=2,help='the weight of loss for positive category')
parser.add_argument('-use_bert',type=bool,default=True,help='whether use bert')
parser.add_argument('-word_embedding_bert',type=bool,default=False,help='whether use bert')
parser.add_argument('-finetune_bert',type=bool,default=False,help='whether finetune bert')
parser.add_argument('-edge_mask',type=str,default='111111',help='edge mask')
parser.add_argument('-conv_type',type=str,default='gcn',help='graph convolution type')

parser.add_argument('-test',type=bool,default=True,help='whether test')
parser.add_argument('-best_model_path',type=str,default='/mnt/nlp-lq/yujunshuai/code/QA/experiments/save_model/qa_graph_gcn_bert/best-validate-model.pt',help='best model path')


args = parser.parse_args()
print(args)

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    train_dataset = QAGraphDataset(root='/mnt/nlp-lq/yujunshuai/code/QA/data',
                             file='processed_train_data.json',
                             tokenizer=tokenizer,
                             is_test=False)
    
    test_dataset = QAGraphDataset(root='/mnt/nlp-lq/yujunshuai/code/QA/data',
                             file='processed_modified_test_data.json',
                             tokenizer=tokenizer,
                             is_test=True)
    
    train_loader = DataListLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataListLoader(test_dataset, batch_size=1, shuffle=False)

    logging.info(f"train data all steps: {len(train_loader)}, "
                    f"test data all steps : {len(test_loader)}")
    
    model = HGCNForTextClassification(num_class=2,
                                      dropout=0.5,
                                      embed_dim=768,
                                      edge_mask=list(map(int, list(args.edge_mask))),
                                      layers=args.gcn_layers,
                                      use_bert=args.use_bert,
                                      finetune_bert=args.finetune_bert,
                                      conv_type=args.conv_type,
                                      word_embedding_bert=args.word_embedding_bert
                                      )
    
    model = model.cuda(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    trainer = Train(model_name=args.model_name,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=args.device,
                    model=model,
                    optimizer=optimizer,
                    epochs=args.epochs,
                    print_step=1,
                    early_stop_patience=args.early_stop_patience,
                    save_model_path=f"./experiments/save_model/{args.model_name}",
                    save_model_every_epoch=False,
                    metric=accuracy_score,
                    num_class=2,
                    tensorboard_path=f'./experiments/tensorboard_log/{args.model_name}')
                    
    if args.test and args.best_model_path:
        print(trainer.test(test_loader, '/mnt/nlp-lq/yujunshuai/code/QA/data/processed_modified_test_data.json', '/mnt/nlp-lq/yujunshuai/code/QA/data/processed_modified_test_data_result.json', args.best_model_path))
    else:
        print(trainer.train())
