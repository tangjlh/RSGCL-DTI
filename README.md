# RSGCL-DTI
RSGCL-DTI: The codes demo for paper "Relational Similarity based Graph Contrastive Learning for DTI prediction".

# Required Packages
View requirements.txt and environment.yml

# Datset
Please make sure dataset format is csv, and columns name are: 'smiles','sequences','label'. We have provided four datasets in the dataset directory. Note: The dataset/BioSNAP directory contains a file named BioSNAP1_5.csv, which is an imbalanced dataset based on BioSNAP data.

If you want to use your own dataset, you need to prepare four files: matrix_d_d.txt(the drug-drug relationship matrix file), matrix_p_p.txt(the protein-protein relationship matrix file), smiles_to_index.pkl(the smiles is the key and the index is the value),  sequence_to_index.pkl(the protein sequence is the key and the index is the value).

# Quick start
You can use the following code to run our demo.<br>
Foe example: python train.py --data_path ./dataset/BioSNAP/BioSNAP.csv --metric auc --dataset_type classification --save_dir save --target_columns label --epochs 150 --ensemble_size 1 --num_folds 10 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds

# Notice
We uploaded the pt file of the model's weights on the equilibrium datasets Human, C.elegans, DrugBank, and BioSNAP datasets to Google Driver
https://drive.google.com/drive/folders/15pz5F-UWmau7I69cYnwVSeYsLKZJrPnm?usp=drive_link

