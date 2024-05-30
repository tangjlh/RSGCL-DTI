from typing import List
import torch
from tqdm import tqdm
from RSGCL.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from RSGCL.models import InteractionModel
import numpy as np
from RSGCL.args import TrainArgs


def predict(model: InteractionModel,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            drug_rep: torch.Tensor,
            pro_rep: torch.Tensor,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            tokenizer=None) -> List[List[float]]:

    model.eval()

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):

        batch: MoleculeDataset
        mol_batch, features_batch, protein_sequence_batch, atom_descriptors_batch, atom_features_batch, \
            bond_features_batch, add_feature, kmer, smiles, drug_index, pro_index = batch.batch_graph(), batch.features(), batch.sequences(),\
            batch.atom_descriptors(), batch.atom_features(), batch.bond_features(), batch.add_features(), \
            batch.three_mer(), batch.smiles(), batch.get_drug_index(), batch.get_protein_index()
        batch_drug_rep, batch_pro_rep = drug_rep[drug_index], pro_rep[pro_index]
        dummy_array = [0]*args.sequence_length
        sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
        new_ar = []
        for arr in sequence_2_ar:
            while len(arr)>args.sequence_length:
                arr.pop(len(arr)-1)
            new_ar.append(np.zeros(args.sequence_length)+np.array(arr))

        sequence_tensor = torch.LongTensor(np.array(new_ar))

        with torch.no_grad():
            batch_preds = model(mol_batch, kmer, smiles, sequence_tensor, batch_drug_rep, batch_pro_rep)
        batch_preds = batch_preds.data.cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
    return preds
