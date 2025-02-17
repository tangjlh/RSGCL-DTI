from .BGRL.transformer import get_graph_drop_transform
from .BGRL.encoder import GCN
from .BGRL.bgrl import BGRL
from .BGRL.scheduler import CosineDecayScheduler
from .BGRL.predictor import MLP_Predictor
from torch_geometric.data import Data
import torch
import numpy as np
import os
from RSGCL.args import TrainArgs
import scipy.sparse as sp
from torch.optim import AdamW
from torch.nn.functional import cosine_similarity
lr_warmup_epochs = 2000
epochs = 2000
lr = 2e-5
mm = 0.99
device = torch.device("cuda:0")

path = TrainArgs().parse_args().data_path.rsplit('/', 1)[0]
numpy_dd = np.loadtxt(path+'/matrix_d_d.txt', dtype=int, delimiter=' ')
edge_index_temp_dd = sp.coo_matrix(numpy_dd)
indices_dd = np.vstack((edge_index_temp_dd.row, edge_index_temp_dd.col))
edge_index_dd = torch.LongTensor(indices_dd)
durg_feature = torch.cat([torch.rand(1, 167) for _ in range(np.size(numpy_dd, 0))], dim=0)
data_dd = Data(x=durg_feature, edge_index=edge_index_dd).cuda()

numpy_pp = np.loadtxt(path+'/matrix_p_p.txt', dtype=int, delimiter=' ')
edge_index_temp_pp = sp.coo_matrix(numpy_pp)
indices_pp = np.vstack((edge_index_temp_pp.row, edge_index_temp_pp.col))
edge_index_pp = torch.LongTensor(indices_pp)
protein_feature = torch.cat([torch.rand(1, 256) for _ in range(np.size(numpy_pp, 0))], dim=0)
data_pp = Data(x=protein_feature, edge_index=edge_index_pp).cuda()

lr_scheduler = CosineDecayScheduler(lr, lr_warmup_epochs, epochs)
mm_scheduler = CosineDecayScheduler(1 - mm, 0, epochs)


def compute_data_representations_only(net, data):
    net.eval()
    with torch.no_grad():
        rep = net(data)
    return rep


def train_dd(step, model_bgrl_dd, optimizer_dd, transform_1, transform_2):
    model_bgrl_dd.train()

    lr = lr_scheduler.get(step)
    for param_group in optimizer_dd.param_groups:
        param_group['lr'] = lr

    mm = 1 - mm_scheduler.get(step)

    optimizer_dd.zero_grad()

    x1, x2 = transform_1(data_dd), transform_2(data_dd)

    q1, y2 = model_bgrl_dd(x1, x2)
    q2, y1 = model_bgrl_dd(x2, x1)

    loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
    loss.backward()

    optimizer_dd.step()
    model_bgrl_dd.update_target_network(mm)
    if step%100 == 0:
        print('drug-drug sim: train/dd_loss', loss, step)


def train_pp(step, model_bgrl_pp, optimizer_pp, transform_1, transform_2):
    model_bgrl_pp.train()

    lr = lr_scheduler.get(step)
    for param_group in optimizer_pp.param_groups:
        param_group['lr'] = lr

    mm = 1 - mm_scheduler.get(step)

    optimizer_pp.zero_grad()

    x1, x2 = transform_1(data_pp), transform_2(data_pp)

    q1, y2 = model_bgrl_pp(x1, x2)
    q2, y1 = model_bgrl_pp(x2, x1)

    loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
    loss.backward()

    optimizer_pp.step()
    model_bgrl_pp.update_target_network(mm)
    if step % 100 == 0:
        print('protein-protein: train/pp_loss', loss, step)


def pre_train(save_dir):
    transform_1 = get_graph_drop_transform(drop_edge_p=0.1, drop_feat_p=0.1)
    transform_2 = get_graph_drop_transform(drop_edge_p=0.2, drop_feat_p=0.2)

    encoder_dd = GCN([167, 256, 300], batchnorm=True)
    predictor_dd = MLP_Predictor(300, 300, hidden_size=256)
    encoder_pp = GCN([256, 256, 300], batchnorm=True)
    predictor_pp = MLP_Predictor(300, 300, hidden_size=256)
    model_bgrl_dd = BGRL(encoder_dd, predictor_dd).to(device)
    model_bgrl_pp = BGRL(encoder_pp, predictor_pp).to(device)
    optimizer_dd = AdamW(model_bgrl_dd.trainable_parameters(), lr=lr, weight_decay=1e-5)
    optimizer_pp = AdamW(model_bgrl_pp.trainable_parameters(), lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        train_dd(epoch, model_bgrl_dd, optimizer_dd, transform_1, transform_2)
        train_pp(epoch, model_bgrl_pp, optimizer_pp, transform_1, transform_2)
    encoder_d = model_bgrl_dd.online_encoder.eval()
    encoder_p = model_bgrl_pp.online_encoder.eval()
    drug_rep = compute_data_representations_only(encoder_d, data_dd)
    pro_reo = compute_data_representations_only(encoder_p, data_pp)
    torch.save(encoder_d.state_dict(), os.path.join(save_dir, 'encoder_drug.pt'))
    torch.save(encoder_p.state_dict(), os.path.join(save_dir, 'encoder_protein.pt'))
    return drug_rep, pro_reo


