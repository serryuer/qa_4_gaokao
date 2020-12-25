import logging
# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)
import random, os, sys, json
import torch
import argparse
from sklearn.metrics import *
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from transformers import AdamW, BertTokenizer

from datasets.QAGraphDataset import QAGraphDataset
from models.HGCN import HGCNForQAClassification
from trainer import Train
from models.BertMatch import BertMatch


parser = argparse.ArgumentParser(description="text gcn with pytorch + torch_geometric")
parser.add_argument('-lr', type=float, default=1e-5, help='initial learning rate [default: 0.001]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 20]')
parser.add_argument('-gcn_layers',type=int,default=3,help='the network layer [default 2]')
parser.add_argument('-hidden_size',type=int,default=512,help='number of hidden size for one rnn cell [default: 512]')
parser.add_argument('-random_seed',type=int,default=1024,help='attention size')
parser.add_argument('-device', type=int, default=3, help='device to use for iterate data, -1 mean cpu,1 mean gpu [default: -1]')
parser.add_argument('-model_name', type=str, default='mr-gcn-layer-3', help='model name')
parser.add_argument('-early_stop_patience', type=int, default=10, help='early stop patience')
parser.add_argument('-loss_weight',type=int,default=2,help='the weight of loss for positive category')
parser.add_argument('-finetune_bert',type=bool,default=False,help='whether finetune bert')

args = parser.parse_args()
print(args)

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    train_dataset = QAGraphDataset(root='./data',
                             file='./data/train_new.csv',
                             tokenizer=tokenizer,
                             is_test=False)
    
    test_dataset = QAGraphDataset(root='./data',
                             file='./data/test_new.csv',
                             tokenizer=tokenizer,
                             is_test=True)
    
    train_loader = DataListLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataListLoader(test_dataset, batch_size=1, shuffle=False)

    logging.info(f"train data all steps: {len(train_loader)}, "
                    f"test data all steps : {len(test_loader)}")

    model = HGCNForQAClassification(num_class=2, 
                                    dropout=args.dropout, 
                                    pretrained_weight=None, 
                                    edge_mask=[0,1,1,1,1], 
                                    use_bert=True,
                                    finetune_bert=args.finetune_bert)
    
    model = BertMatch()
    
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
                    epochs=12,
                    print_step=1,
                    early_stop_patience=3,
                    save_model_path=f"./experiments/save_model/{args.model_name}",
                    save_model_every_epoch=False,
                    metric=accuracy_score,
                    num_class=2,
                    tensorboard_path=f'./experiments/tensorboard_log/{args.model_name}')
                    # model_cache='./experiments/save_model/mr-gcn-layer-3/best-validate-model.pt')
    # print(trainer.eval())
    print(trainer.train())
    # print(trainer.test('./data/test.csv', './data/test_res.csv'))
    # print(trainer.test(test_loader, './data/test_new.csv', './data/test_new_res.csv'))
