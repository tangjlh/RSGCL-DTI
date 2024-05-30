#!/bin/bash 
echo "Please enter the GPU address"
read GPU
echo "Sellected GPU: $GPU"

#train
CUDA_VISIBLE_DEVICES=$GPU nohup python train.py --data_path '' --metric auc --dataset_type classification --save_dir '' --target_columns label --epochs 1 --ensemble_size 1 --num_folds 1 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds >& out_name.out &