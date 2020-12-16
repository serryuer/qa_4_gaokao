# QA classification

# 基于规则的方法

| precision	| recall |
|  ----  | ----  | ----  |
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

### GAT
| config | precision | recall |
|  ----  | ----  | ----  |
| GAT | 0.333 | 0.524 |
| GAT+bert | * | * | 
| GAT+bert finetune | * | * | 


### HGAT

### HGAT + attention_heads
| config | precision | recall |
|  ----  | ----  | ----  |
| ** | * | * | 
| ** | * | * | 
| ** | * | * | 
| ** | * | * |
