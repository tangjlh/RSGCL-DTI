import os
from typing import List, Union, Tuple
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from .dmpnn import MPN
from RSGCL.args import TrainArgs
from RSGCL.features import BatchMolGraph
from RSGCL.nn_utils import initialize_weights

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# python train.py --data_path ./dataset/biosnap.csv --metric auc --dataset_type classification --save_dir GCL_and_Seq_smiles_thredhold0.5 --target_columns label --epochs 150 --ensemble_size 1 --num_folds 10 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds


class InteractionModel(nn.Module):

    def __init__(self, args: TrainArgs, featurizer: bool = False):

        super(InteractionModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        self.embedding_xt = nn.Embedding(args.vocab_size, args.prot_hidden*2)
        self.conv_in = nn.Conv1d(in_channels=args.sequence_length, out_channels=args.prot_1d_out, kernel_size=1)

        self.convs = nn.ModuleList(
            [nn.Conv1d(args.prot_hidden, 2 * args.prot_hidden, args.kernel_size, padding=args.kernel_size // 2) for _ in
             range(args.prot_1dcnn_num)])
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(args.prot_1d_out)
        self.attention = nn.MultiheadAttention(args.hidden_size, 1)
        self.fc1 = nn.Linear(1200, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)
        self.d3 = nn.Dropout(p=0.1)
        self.leaky = nn.LeakyReLU()
        self.prot_1d_out = args.prot_1d_out
        self.embedding_1mer = nn.Embedding(args.one_mer_classes, args.prot_hidden * 2)
        self.embedding_2mer = nn.Embedding(args.two_mer_classes, args.prot_hidden * 2)
        self.embedding_3mer = nn.Embedding(args.three_mer_classes, args.prot_hidden * 2)
        self.CNN_kmer = nn.Sequential(
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=args.hidden_size, kernel_size=16),
            nn.ReLU()
        )
        self.embedding_smiles = nn.Embedding(args.smiles_element_classes, args.prot_hidden * 2)
        self.CNN_smiles = nn.Sequential(
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=args.hidden_size, kernel_size=4),
            nn.ReLU()
        )
        self.CNN_SEQ = nn.Sequential(
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=args.hidden_size, kernel_size=4),
            nn.ReLU()
        )
        self.ffn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=args.prot_hidden * 2, out_channels=512, kernel_size=4)
        )

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        self.encoder = MPN(args)
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[
                        BatchMolGraph]],
                    features_batch: List[np.ndarray] = None,
                    atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        return self.encoder(batch, features_batch, atom_descriptors_batch)

    def forward(self,
                batch: Union[
                    List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                kmer_tensor: torch.Tensor = None,
                smiles_tensor: torch.Tensor = None,
                sequence_tensor: List[np.ndarray] = None,
                drug_rep: torch.Tensor = None,
                pro_rep: torch.Tensor = None
                ):

        sequence_tensor = sequence_tensor.cuda()
        embedded_xt = self.embedding_xt(sequence_tensor).permute(0,2,1)
        protein_tensor = self.CNN_SEQ(embedded_xt)

        mpnn_out = self.encoder(batch)  # [B, N, C]

        drug_maxpool = nn.MaxPool1d(mpnn_out.size(1))
        seq_maxpool = nn.MaxPool1d(protein_tensor.size(2))

        drug = drug_maxpool(mpnn_out.permute(0,2,1)).squeeze(2)
        seq = seq_maxpool(protein_tensor).squeeze(2)

        protein_res = torch.cat((seq, pro_rep), dim=1)
        drug_res = torch.cat((drug, drug_rep), dim=1)

        pair = torch.cat((protein_res, drug_res), dim=1)

        f1 = self.d2(self.leaky(self.fc1(self.d1(pair))))
        f2 = self.d3(self.leaky(self.fc2(f1)))
        f3 = self.leaky(self.fc3(f2))
        output = self.fc4(f3)

        return output


