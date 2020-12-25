# lr 
nohup python hgcn_main.py \
    -lr 1e-6 \
    -gcn_layers 2 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_lr1e6 >> ./log/qa_graph_lr1e6.log 2>&1 &
sleep 3

nohup python hgcn_main.py \
    -lr 5e-6 \
    -gcn_layers 2 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_lr5e6 >> ./log/qa_graph_lr5e6.log 2>&1 & 
sleep 3

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_lr1e5 >> ./log/qa_graph_lr1e5.log 2>&1 &
sleep 3

nohup python hgcn_main.py \
    -lr 5e-5 \
    -gcn_layers 2 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_lr5e5 >> ./log/qa_graph_lr5e5.log 2>&1 &
sleep 3

nohup python hgcn_main.py \
    -lr 1e-4 \
    -gcn_layers 2 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_lr1e4 >> ./log/qa_graph_lr1e4.log 2>&1 &
sleep 3

# layer 
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 1 \
    -device 1 \
    -conv_type gcn \
    -model_name qa_graph_layer1 >> ./log/qa_graph_gcn_layer1.log 2>&1 &
sleep 3

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 1 \
    -conv_type gcn \
    -model_name qa_graph_layer2 >> ./log/qa_graph_gcn_layer2.log 2>&1 &
sleep 3

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 3 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_layer3 >> ./log/qa_graph_gcn_layer3.log 2>&1 &
sleep 3

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 4 \
    -device 3 \
    -conv_type gcn \
    -model_name qa_graph_layer4 >> ./log/qa_graph_gcn_layer4.log 2>&1 &
sleep 3


# edge
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 6 \
    -model_name qa_graph_wof2s \
    -conv_type gcn \
    -edge_mask 111110 >> ./log/qa_graph_wof2s.log 2>&1 &


nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 6 \
    -model_name qa_graph_wosq \
    -conv_type gcn \
    -edge_mask 111001 >> ./log/qa_graph_wosq.log 2>&1 &


# bert
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 4 \
    -model_name qa_graph_gcn_bert \
    -conv_type gcn \
    -use_bert True >> ./log/qa_graph_gcn_bert.log 2>&1 &

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 4 \
    -model_name qa_graph_gcn_bert_finetune \
    -conv_type gcn \
    -finetune_bert True \
    -use_bert True >> ./log/qa_graph_gcn_bert_finetune.log 2>&1 &

# gat
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 1 \
    -model_name qa_graph_gat \
    -conv_type gat >> ./log/qa_graph_gat.log 2>&1 &

# gat layer
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 3 \
    -device 1 \
    -model_name qa_graph_gat \
    -conv_type gat >> ./log/qa_graph_gat_layer3.log 2>&1 &

# gat bert
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 1 \
    -model_name qa_graph_gat \
    -conv_type gat \
    -use_bert True >> ./log/qa_graph_gat_bert.log 2>&1 &

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 4 \
    -model_name qa_graph_gat \
    -finetune_bert True \
    -use_bert True \
    -conv_type gat > ./log/qa_graph_gat_bert_finetune.log 2>&1 &


# hgat

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 7 \
    -model_name qa_graph_gat \
    -conv_type hgat >> ./log/qa_graph_hgat.log 2>&1 &

# hgat layer 
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 3 \
    -device 6 \
    -model_name qa_graph_gat \
    -conv_type hgat >> ./log/qa_graph_hgat_layer3.log 2>&1 &


# hgat bert

nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 7 \
    -model_name qa_graph_gat \
    -use_bert True \
    -conv_type hgat >> ./log/qa_graph_hgat_bert.log 2>&1 &
    
nohup python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 2 \
    -model_name qa_graph_gat \
    -finetune_bert True \
    -use_bert True \
    -conv_type hgat > ./log/qa_graph_hgat_bert_finetune.log 2>&1 &

# test
python hgcn_main.py \
    -lr 1e-5 \
    -gcn_layers 2 \
    -device 4 \
    -model_name qa_graph_gcn_bert \
    -conv_type gcn \
    -use_bert True \
    -test True \
    -best_model_path ./experiments/save_model/qa_graph_gcn_bert/best-validate-model.pt
