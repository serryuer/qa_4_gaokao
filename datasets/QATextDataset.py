import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class QATextDataset(Dataset):
    def __init__(self,
                 root,
                 file, 
                 tokenizer,
                 w2v = None,
                 is_test = False,
                 max_sequence_length=256):
        self.root = root
        self.max_sequence_length = max_sequence_length
        self.data_file = file
        self.tokenizer = tokenizer
        self.is_test = is_test
        
        self._init_data()

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
                self,
                tokens,
                input_ids,
                input_mask,
                segment_ids,
        ):
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids

    def convert_sentence_to_features(self, sentence):
        max_sequence_length = self.max_sequence_length + 2
        tokenize_result = self.tokenizer.tokenize(sentence)

        # truncate sequences pair
        while len(tokenize_result) > self.max_sequence_length:
            tokenize_result.pop()

        tokens = []
        tokens.append('[CLS]')
        segment_ids = []
        segment_ids.append(0)

        for token in tokenize_result:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length
        assert len(segment_ids) == max_sequence_length

        return QATextDataset.BertInputFeatures(
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )
        
    def construct_sample(self, sample):
        question = sample[0]['question']
        sents = [item['sentence'] for item in sample]
        labels = [int(item['label']) for item in sample]
        input_features = [self.convert_sentence_to_features(sent) for sent in [question] + sents]
        input_ids = [input_feature.input_ids for input_feature in input_features]
        input_mask = [input_feature.input_mask for input_feature in input_features]
        segment_ids = [input_feature.segment_ids for input_feature in input_features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, input_mask, segment_ids, labels

    def _init_data(self):
        processed_save_path = os.path.join(self.root, 'processed_text_train.pkl' if not self.is_test else 'processed_text_dev.pkl')
        if os.path.exists(processed_save_path):
            self.data_list = torch.load(processed_save_path)
            return 
        self.data_list = []
        with open(os.path.join(self.root, self.data_file)) as f:
            sample = []
            for line in tqdm(f, desc=f"processed file {self.data_file}"):
                id, question, sentence, label = [item.strip() for item in line.strip().split('\t')[:4]]
                if not sample or sample[0]['question'] == question:
                    sample.append({'sentence': sentence, 'question': question, 'label':int(label)})
                else:
                    data = self.construct_sample(sample)
                    self.data_list.append(data)
                    sample = [{'sentence': sentence, 'question': question, 'label':int(label)}]
            data = self.construct_sample(sample)
            self.data_list.append(data)
        print(f"data length : {len(self.data_list)}")
        torch.save(self.data_list, processed_save_path)

    def __getitem__(self, item):
        return [data.long() for data in self.data_list[item]]

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    dataset = QATextDataset(
        root = '/mnt/nlp-lq/yujunshuai/code/QA/data/',
        file = '/mnt/nlp-lq/yujunshuai/code/QA/data/train_new.csv',
        tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[0])
