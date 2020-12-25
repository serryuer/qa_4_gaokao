# QA classification

# Get Started

```
# 1. 原始数据格式转化成json，生成data/train_data.json、data/test_data.json
python datasets/data_prepare.py

# 2. 基于规则计算每个句子的属性特征以及基础统计特征
python datasets/cal_statistics.py

# 3. 模型训练
# 目前最优模型
python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 5 \
    -model_name qa_graph_gcn_bert \
    -conv_type gcn \
    -use_bert True

# 4. 记载模型并测试
python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 3 \
    -model_name qa_graph_gcn_bert \
    -conv_type gcn \
    -use_bert True \
    -test True \
    -best_model_path ./experiments/save_model/qa_graph_gcn_bert/best-validate-model.pt

```

# 基于规则的方法

| precision	| recall |
|  ----  | ----  |
| 0.33 | 0.50 | 
| 0.33 | 0.50 | 
| 0.33 | 0.67 | 
| 0.17 | 0.50 | 
| 0.33 | 0.33 | 
| 0.17 | 0.33 | 
| 0.33 | 0.50 | 
| 1.00 | 0.86 | 
| 0.67 | 0.57 | 
| 0.67 | 0.67 | 
| **0.43** | **0.54** | 

# 规则+GCN/GAT/HGAT

## GCN

### learning_rate

| config | precision | recall |
|  ----  | ----  | ----  |
| 1e-6 | 0.317 | 0.428 | 
| 5e-6 | 0.367 | 0.510 | 
| **1e-5** | **0.367** | **0.520** | 
| 5e-5 | 0.350 | 0.493 |

### layer

| config | precision | recall |
|  ----  | ----  | ----  |
| 1 | 0.317 | 0.449 | 
| **2** | **0.367** | **0.520** | 
| 3 | 0.350 | 0.493 | 
| 4 | 0.350 | 0.518 |

### edge types
| config | precision | recall |
|  ----  | ----  | ----  | 
| **w2w + w2sq + sq2w + s2q + q2s + f2s** | **0.367** | **0.520** | 
| without f2s | 0.233 | 0.360 | 
| without s2q  q2s | 0.367 | 0.518 | 

### bert
| config | precision | recall |
|  ----  | ----  | ----  | 
| lstm | 0.367 | 0.520 |
| sent_bert | 0.400 | 0.578 | 
| sent_bert + word_bert | * | * | 
| sent_bert_finetune | * | * | 
| sent_bert_finetune + word_bert_finetune | * | * | 
