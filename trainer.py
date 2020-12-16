import logging
import os
import time
import json
import numpy as np
import torch
from sklearn.metrics import *
# log format
from tensorboardX import SummaryWriter
from tqdm import tqdm


C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)


class Train(object):
    def __init__(self, 
                 device,
                 model_name, 
                 train_loader, 
                 test_loader, 
                 model, 
                 optimizer, 
                 epochs, 
                 print_step,
                 early_stop_patience, 
                 save_model_path, 
                 num_class, 
                 save_model_every_epoch=False,
                 metric=f1_score, 
                 tensorboard_path=None,
                 model_cache=None):
        self.device = device
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.print_step = print_step
        self.early_stop_patience = early_stop_patience
        self.save_model_every_epoch = save_model_every_epoch
        self.save_model_path = save_model_path
        self.metric = metric
        self.num_class = num_class

        self.tensorboard_path = tensorboard_path

        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)
        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.best_val_epoch = 0
        self.best_val_score = 0

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path)
        
        if model_cache:
            self.model.load_state_dict(torch.load(model_cache))

    def _save_model(self, model_name):
        torch.save(self.model.state_dict(), os.path.join(self.save_model_path, model_name + '.pt'))

    def _early_stop(self, epoch, score):
        if score > self.best_val_score:
            self.best_val_score = score
            self.best_val_epoch = epoch
            self._save_model('best-validate-model')
        else:
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + self.model_name + f"Validate has not promote {epoch - self.best_val_epoch}/{self.early_stop_patience}")
            if epoch - self.best_val_epoch > self.early_stop_patience:
                logging.info(self.model_name + f"-epoch {epoch}" + ":"
                             + f"Early Stop Train, best score locate on {self.best_val_epoch}, "
                             f"the best score is {self.best_val_score}")
                return True
        return False

    def eval(self):
        logging.info(self.model_name + ":" + "## Start to evaluate. ##")
        self.model.eval()
        eval_loss = 0.0
        total_precision = []
        total_recall = []
        self.model.eval()
        for batch_count, batch_data in enumerate(self.test_loader):
            with torch.no_grad():
                outputs = self.model([data.to(self.device) for data in batch_data])
                tmp_eval_loss, logit = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                logits = logit.detach().cpu().numpy()[:, 1]
                if self.model_name.find('graph') != -1:
                    precision, recall = outputs[2:]
                else:
                    precision, recall = self._cal_metrics(batch_data[-1].squeeze(0).detach().cpu().numpy(), logits)
                total_precision.append(precision)
                total_recall.append(recall)
        result = {}
        result['precision'] = sum(total_precision)/(batch_count + 1)
        result['recall'] = sum(total_recall)/(batch_count + 1)
        result['f1'] = 0 if precision == 0 else (2 * (sum(total_precision)/(batch_count + 1)) * sum(total_recall)/(batch_count + 1)) / (sum(total_precision)/(batch_count + 1) + sum(total_recall)/(batch_count + 1))
        return result
    
    def _cal_metrics(self, label, logits):
        # logits = logits.detach().cpu().numpy()[:, 1]
        sorted_logits = np.argsort(-logits)
        tp = sum([1 if label[i] == 1 else 0 for i in sorted_logits[:6]])
        precision = 0 if tp == 0 else tp / 6
        recall = 0 if tp == 0 else tp / sum(label)
        return precision, recall

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_precision = []
            total_recall = []
            tr_loss = 0.0
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"## The {epoch} Epoch, all {self.epochs} Epochs ! ##")
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"The current learning rate is {self.optimizer.param_groups[0].get('lr')}")
            self.model.train()
            since = time.time()
            for batch_count, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model([data.to(self.device) for data in batch_data])
                loss, logit = outputs[:2]
                loss.backward()
                self.optimizer.step()
                logits = logit.detach().cpu().numpy()[:, 1]
                tr_loss += loss.mean().item()
                if self.model_name.find('graph') != -1:
                    precision, recall = outputs[2:]
                else:
                    precision, recall = self._cal_metrics(batch_data[-1].squeeze(0).detach().cpu().numpy(), logits)
                total_precision.append(precision)
                total_recall.append(recall)
                if (batch_count + 1) % self.print_step == 0:
                    logging.info(self.model_name + f"-epoch {epoch}" + ":" + f"batch {batch_count + 1} : loss is {tr_loss/(batch_count + 1)},"
                                 f"precision is {sum(total_precision)/(batch_count + 1)}, "
                                 f"recall is {sum(total_recall)/(batch_count + 1)}, "
                                 f"f1 is {0 if precision == 0 else (2 * (sum(total_precision)/(batch_count + 1)) * sum(total_recall)/(batch_count + 1)) / (sum(total_precision)/(batch_count + 1) + sum(total_recall)/(batch_count + 1))}")
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_loss', tr_loss / (batch_count + 1),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/precision',
                                              sum(total_precision)/(batch_count + 1),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_recall',
                                              sum(total_recall)/(batch_count + 1),
                                              batch_count + epoch * len(self.train_loader))
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_f1', 
                                              0 if precision == 0 else (2 * (sum(total_precision)/(batch_count + 1)) * sum(total_recall)/(batch_count + 1)) / (sum(total_precision)/(batch_count + 1) + sum(total_recall)/(batch_count + 1)),
                                              batch_count + epoch * len(self.train_loader))

            val_score = self.eval()
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_precision', val_score['precision'], epoch)
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_recall', val_score['recall'], epoch)
            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_f1', val_score['f1'], epoch)

            logging.info(self.model_name + ": epoch" +
                         f"Epoch {epoch} Finished with time {format(time.time() - since)}, " +
                         f"validate score {val_score}")
            if self.save_model_every_epoch:
                self._save_model(f"{self.model_name}-{epoch}-{val_score['precision']}")
                
            res = {}
            res['validation_precision'] = val_score['precision']
            res['validation_recall'] = val_score['recall']
            res['train_precision'] = sum(total_recall)/(batch_count + 1)
            res['train_recall'] = sum(total_recall)/(batch_count + 1)
            with open(os.path.join(self.save_model_path, f'metrics_epoch_{epoch}.json'), mode='w') as f:
                f.write(json.dumps(res) + '\n')
            if self._early_stop(epoch, val_score['recall']):
                break
        self.tb_writer.close()
        
    def test(self, data_loader, input_file, output_file, best_model_path):
        logging.info(self.model_name + ":" + "## Start to test. ##")
        self.model.eval()
        self.model.load_state_dict(torch.load(best_model_path))
        predicts = []
        logits = []
        total_precision = []
        total_recall = []
        for batch_count, batch_data in enumerate(data_loader):
            with torch.no_grad():
                outputs = self.model([data.to(self.device) for data in batch_data])
                _, logit, precision, recall = outputs
                logits = logit.detach().cpu().numpy()[:, 1]
                sorted_logits = np.argsort(-logits)
                predicts.extend([1 if i in sorted_logits[:6] else 0 for i in range(len(sorted_logits))])
                total_precision.append(precision)
                total_recall.append(recall)
        result = {}
        result['precision'] = sum(total_precision)/(batch_count + 1)
        result['recall'] = sum(total_recall)/(batch_count + 1)
        result['f1'] = 0 if precision == 0 else (2 * (sum(total_precision)/(batch_count + 1)) * sum(total_recall)/(batch_count + 1)) / (sum(total_precision)/(batch_count + 1) + sum(total_recall)/(batch_count + 1))
        
        with open(input_file, 'r') as f, open(output_file, 'w') as f1:
            start = 0
            f1.write(json.dumps(result) + '\n')
            print(json.dumps(result))
            for i, line in enumerate(f):
                data = json.loads(line)
                for sent, label, pred in zip([item[0] for item in data['sentences']], data['labels'], predicts[start:start+len(data['sentences'])]):
                    print(f"{data['question']}\t{sent}\t{label}\t{pred}")
                    f1.write(f"{data['question']}\t{sent}\t{label}\t{pred} + \n")
                start += len(data['sentences'])
        return result