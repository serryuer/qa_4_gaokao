import os, sys, json


labels = []
with open('/mnt/nlp-lq/yujunshuai/code/QA/data/modified_test.csv') as f:
    for line in f:
        labels.append(1 if line.strip()[-2:] == '11' else 0)

with open('/mnt/nlp-lq/yujunshuai/code/QA/data/processed_test_data.json') as f, open('/mnt/nlp-lq/yujunshuai/code/QA/data/processed_modified_test_data.json', 'w') as f1:
    start = 0
    for line in f:
        data = json.loads(line)
        old_label = data['labels']
        new_label = labels[start : start+len(data['labels'])]
        data['labels'] = [1 if new_label[i] == 1 else old_label[i] for i in range(len(old_label))]
        start += len(data['labels'])
        f1.write(json.dumps(data, ensure_ascii=False) + '\n')
        
            