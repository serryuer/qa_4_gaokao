import json, os, random, re
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict
from math import log
import scipy.sparse as sp

class QAGraphDataset(InMemoryDataset):
    def __init__(self, 
                 root,
                 file, 
                 tokenizer,
                 is_test = False,
                 max_sequence_length=256):
        self.root = root
        self.max_sequence_length = max_sequence_length
        self.data_file = file
        self.tokenizer = tokenizer
        self.is_test = is_test
        
        super(QAGraphDataset, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.is_test:
            return 'qa_test.dataset'
        else:
            return 'qa_train.dataset'

    def download(self):
        pass
 
    def get(self, idx):
        return self.data[idx]
    
    def len(self):
        return len(self.data)
    
    def construct_graph(self, sample):
        s2q_edge_index = [[], []]
        q2s_edge_index = [[], []]
        sq2w_edge_index = [[], []]
        sq2w_edge_attr = []
        w2sq_edge_index = [[], []]
        w2sq_edge_attr = []
        f2s_edge_index = [[], []]
        w2w_edge_index = [[], []]
        w2w_edge_attr = []
        node_features = []
        node_labels = []
        # 0 sent, 1 feature, 2 word
        node_types = []
        
        vocab = []
        for s in [sample['question']] + [item[0] for item in sample['sentences']]:
            words = self.tokenizer.tokenize(s)
            vocab.extend(words)
            tokens = ['[CLS]']
            tokens.extend(words)
            tokens = tokens[:self.max_sequence_length - 1]
            tokens.append('[SEP]')
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            while len(input_ids) < self.max_sequence_length:
                input_ids.append(0)
            assert len(input_ids) == self.max_sequence_length
            node_features.append(input_ids)
            node_types.append(0)
        for sent in sample['sentences']:
            vector = sent[1:]
            vector.extend([0] * (self.max_sequence_length - len(vector)))
            node_features.append(vector)
            node_types.append(1)
        vocab = list(set(vocab))
        word_id_map = {word : i for i, word in enumerate(vocab)}
        for word, id in zip(vocab, self.tokenizer.convert_tokens_to_ids(vocab)):
            vector = [0] * self.max_sequence_length
            vector[0] = id
            node_features.append(vector)
            node_types.append(2)
        
        for i in range(len(sample['sentences'])):
            s2q_edge_index[0].append(i + 1)
            s2q_edge_index[1].append(0)
            q2s_edge_index[0].append(0)
            q2s_edge_index[1].append(i + 1)
            f2s_edge_index[0].append(len(sample['sentences']) + i + 1)
            f2s_edge_index[1].append(i + 1)
        
        node_labels = [-1] + list(map(int, sample['labels'])) + [-1] * (len(sample['sentences']) + len(vocab))
        
        label_mask = torch.tensor(np.array([0] + [1] * len(sample['sentences']) + [0] * (len(vocab) + len(sample['sentences'])), dtype=np.bool))
        
        all_sents = [sample['question']] + [item[0] for item in sample['sentences']]
        # construct word to word edge
        window_size = 10
        windows = []
        for sent in all_sents:
            words = self.tokenizer.tokenize(sent)
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
                    
        word_window_freq = defaultdict(int)
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                word_window_freq[window[i]] += 1
                appeared.add(window[i])

        word_pair_count = defaultdict(int)
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i_id = word_id_map[window[i]]
                    word_j_id = word_id_map[window[j]]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_count[str(word_i_id) + ',' + str(word_j_id)] += 1
                    word_pair_count[str(word_j_id) + ',' + str(word_i_id)] += 1
        
        # pmi as weights
        num_window = len(windows)
        for key in word_pair_count:
            i, j = list(map(int, key.split(',')))
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) /
                    (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            w2w_edge_index[0].append(i + 2 * len(all_sents) - 1)
            w2w_edge_index[1].append(j + 2 * len(all_sents) - 1)
            w2w_edge_attr.append(pmi)
        
        doc_word_freq = defaultdict(int)
        for i, sent in enumerate(all_sents):
            words = self.tokenizer.tokenize(sent)
            for word in words:
                word_id = word_id_map[word]
                doc_word_freq[str(i) + ',' + str(word_id)] += 1
                
        word_doc_list = defaultdict(list)
        for i, sent in enumerate(all_sents):
            words = self.tokenizer.tokenize(sent)
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                word_doc_list[word].append(i)
                appeared.add(word)
        word_doc_freq = {word: len(word_doc_list[word]) for word in word_doc_list}

        for i, sent in enumerate(all_sents):
            words = self.tokenizer.tokenize(sent)
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                w2sq_edge_index[0].append(2 * len(all_sents) - 1 + j)
                w2sq_edge_index[1].append(i)
                idf = log(1.0 * (len(all_sents) + 1) /
                        word_doc_freq[vocab[j]])
                w2sq_edge_attr.append(freq * idf)
                doc_word_set.add(word)
        
        sq2w_edge_index[0] = deepcopy(w2sq_edge_index[1])
        sq2w_edge_index[1] = deepcopy(w2sq_edge_index[0])
        sq2w_edge_attr = deepcopy(w2sq_edge_attr)

        def format_adj(adj):
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
            return adj

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        adj = sp.csr_matrix((w2w_edge_attr, (w2w_edge_index[0], w2w_edge_index[1])), shape=(len(node_types), len(node_types)))
        adj = format_adj(adj)

        if isinstance(adj, list):
            for i in range(len(adj)):
                adj[i] = to_tuple(adj[i])
        else:
            adj = to_tuple(adj)

        w2w_edge_index = adj[0].transpose()
        w2w_edge_attr = adj[1]
    
        w2w_edge_index = np.array(w2w_edge_index)
        sq2w_edge_index = np.array(sq2w_edge_index)
        w2sq_edge_index = np.array(w2sq_edge_index)
        q2s_edge_index = np.array(q2s_edge_index)
        s2q_edge_index = np.array(s2q_edge_index)
        f2s_edge_index = np.array(f2s_edge_index)
        
        data = Data(x=torch.LongTensor(node_features),
                    node_type=torch.LongTensor(node_types),
                    w2w_edge_index=torch.LongTensor(w2w_edge_index),
                    w2w_edge_attr=torch.FloatTensor(w2w_edge_attr),
                    w2sq_edge_index=torch.LongTensor(w2sq_edge_index),
                    w2sq_edge_attr=torch.FloatTensor(w2sq_edge_attr),
                    sq2w_edge_index=torch.LongTensor(sq2w_edge_index),
                    sq2w_edge_attr=torch.FloatTensor(sq2w_edge_attr),
                    s2q_edge_index=torch.LongTensor(s2q_edge_index),
                    q2s_edge_index=torch.LongTensor(q2s_edge_index),
                    f2s_edge_index=torch.LongTensor(f2s_edge_index),
                    label_mask=torch.tensor(label_mask),
                    y=torch.LongTensor(node_labels))
        
        return data

    def process(self):
        # Read data_utils into huge `Data` list.
        data_list = []
        with open(os.path.join(self.root, self.data_file)) as f:
            for line in tqdm(f, desc=f"processed file {self.data_file}"):
                sample = json.loads(line)
                data = self.construct_graph(sample)
                data_list.append(data)
        print(f"data length : {len(data_list)}")
        torch.save(data_list, self.processed_paths[0])

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    dataset = QAGraphDataset(
        root = '/mnt/nlp-lq/yujunshuai/code/QA/data/',
        file = 'processed_train_data.json',
        tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[0])
