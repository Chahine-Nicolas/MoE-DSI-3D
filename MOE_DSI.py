from scipy.spatial.distance import cdist
import os
import sys
import glob
import random
import numpy as np
import logging
import json
import torch
from torch import nn
import math
#from pathlib import Path
import matplotlib.pyplot as plt
#####################################################################################
# Load poses
# ####################################################################################
import time

from mixture_of_experts import MoE

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

def _pad_tensors_to_max_len( tensor, max_length,tokenizer):
    # If PAD token is not defined at least EOS token has to be defined
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    tensor[tensor == -100] = tokenizer.pad_token_id
    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor


def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    return Tcam @ cam2velo


def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)

class Timer(object):
    """A simple timer."""
    # Ref: https://github.com/chrischoy/FCGF/blob/master/lib/timer.py

    def __init__(self, binary_fn=None, init_val=0):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.binary_fn = binary_fn
        self.tmp = init_val

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0

    @property
    def avg(self):
        return self.total_time / self.calls

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if self.binary_fn:
            self.tmp = self.binary_fn(self.tmp, self.diff)
        if average:
            return self.avg
        else:
            return self.diff

""
def moe_training(model, eval_subset, eval_set, eval_indices, eval_loader, data_collator, tokenizer, cfg, checkpoint_dir, checkp_to_eval, prefix_dict, ID_MAX_LENGTH=10):
    print("eval lognet")


    save_descriptors = False
    save_counts = False
    plot_pr_curve = True
    
    eval_seq=cfg['DATA_CONFIG']['SEQ']
    if 'Kitti' in cfg['DATA_CONFIG']['DATASET']:
        log3dnet_dir=os.getenv('LOG3DNET_DIR')
        revisit_criteria=3
        not_revisit_criteria=20
        skip_time=30
        revisit_json_file = 'is_revisit_D-{}_T-{}_v2.json'.format(
            int(revisit_criteria), int(skip_time))
        cd_thresh_min=0.001
        cd_thresh_max=5 # au lieu de 1
        num_thresholds=5000
        num_beams = 10
        ## ==== Kitti =====
        print("kitti dataset")
        kitti_dir= os.getenv('WORKSF') + '/datas/datasets/'
        sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(sequence_path + 'poses.txt')
    
        min_bbox = np.min(positions_database,0) 
        positions_database = positions_database - min_bbox
        
        revisit_json_dir = os.path.join(os.path.dirname(__file__), '/config/kitti_tuples/')
        revisit_json = json.load(open(log3dnet_dir + revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]

    elif cfg['DATA_CONFIG']['DATASET'] == "LHD_dataset" :
        revisit_criteria=1
        not_revisit_criteria=60
        
        cd_thresh_min=0.001
        cd_thresh_max=5 # au lieu de 1
        num_thresholds=5000
        num_beams = 10
        
        sequence_path = cfg['DATA_CONFIG']['DATA_PATH'] + "/"
        with open(sequence_path + "poses_grid2.json") as f: 
            positions_database = json.load(f)


    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    num_correct_loc_all = 0
    hit_at_10 = 0
    dictio = []
    dictio_to_save = []
    print("Start looop")

    checkpoint_paths = ["/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_lhd_long_indx_shuffle/checkpoint-1000/pytorch_model.bin","/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_lhd_long_indx_zone_B0_1m/checkpoint-4300/pytorch_model.bin"]
    transformer_list = []
    
    for chkp in checkpoint_paths:
        state_dict = torch.load(chkp)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        transformer_list.append(model)

    class TransformerExperts(nn.Module):
        def __init__(self, transformer_list):
            super().__init__()
            self.experts = nn.ModuleList(transformer_list)  # each is a full model
    
        def forward(self, x, expert_indices):
            # x: [batch, tokens, dim]
            # expert_indices: [batch, tokens] – tells which expert each token goes to
            batch, tokens, dim = x.shape
            out = torch.zeros_like(x)
    
            for i, expert in enumerate(self.experts):
                mask = (expert_indices == i)  # shape [batch, tokens]
                if mask.any():
                    masked_x = x[mask]  # shape [num_selected, dim]
                    output = expert(masked_x.unsqueeze(1)).squeeze(1)  # or adapt to input shape
                    out[mask] = output
            return out

    experts = TransformerExperts(transformer_list)
    import pdb; pdb.set_trace()

    class ModelWithMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = moe
            self.num_classes = 2
            self.classifier = nn.Linear(512, self.num_classes)
    
        def forward(self, x):
            # x shape: [batch, seq_len, 512]
            out, aux_loss = self.moe(x)
            logits = self.classifier(out)
            return logits, aux_loss

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_moe = ModelWithMoE().to(device)
    optimizer = torch.optim.Adam(model_moe.parameters(), lr=1e-4)
    
    import pdb; pdb.set_trace()

    # input_data = data_collator(torch.utils.data.Subset(eval_subset,range(0, 0+1))) 
    # logits, aux_loss = model_moe(input_data)

    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
            optimizer.zero_grad()
            logits, aux_loss = model(x_batch)
    
            task_loss = nn.CrossEntropyLoss()(logits.view(-1, num_classes), y_batch.view(-1))
            total_loss = task_loss + moe.loss_coef * aux_loss
    
            total_loss.backward()
            optimizer.step()


    
        # moe 





















    


