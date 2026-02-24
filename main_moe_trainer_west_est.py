import _init_path
import argparse
import datetime
import glob

# Avoid tokenizers parallelism fork warning
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from extern.log3dnet.SOP import SOP
from collections import Counter
from dataclasses import replace 
import time
# 
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP

import hostlist

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter 
import copy 
import traceback
import logging

from extern.pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from extern.pcdet.datasets import build_dataloader
from extern.pcdet.utils import common_utils
#from extern.train_utils.optimization import build_optimizer, build_scheduler
#from extern.train_utils.train_utils import train_model
import numpy as np

## Blip2
import requests
from PIL import Image
from transformers import AutoProcessor,AutoModel, AutoConfig, AutoTokenizer, TrainingArguments , HfArgumentParser
from extern.blip2.modeling_blip_2 import Blip2ModelQuerryLearning
from extern.blip2.processing_blip_2 import Blip2Processor
from transformers import BertTokenizer, BertModel,BertLMHeadModel,MT5Tokenizer
from extern.blip2.modeling_bert_generation   import BertGenerationDecoder
from transformers import  GPTQConfig

## DSI QG
from dataclasses import dataclass
from transformers.trainer import Trainer
from transformers import PreTrainedTokenizer, DataCollatorWithPadding,PretrainedConfig
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import   MT5ForConditionalGeneration
from extern.git.modeling_git import GitModel,GitForCausalLM

from evaluate_moe_west_est import eval_log3dnet
#from evaluate_overfit import eval_overfit
#from compute_hierarchical_index import compute_hierarchical_clustering

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import TrainerCallback

##################################
# read pos
# #################################
from module_loader_kitti_pose import * # add for more metrics
import math
import gc

WORK_PATH = os.getenv('WORKSF')

## ====  Usefull stuff =======
def forward_nan_hook(self, inp, output):
    print("not implemented yet")

def backward_nan_hook(name):
    def hook(module, grad_input, grad_output):
        if (len(grad_input) == 0 or len(grad_output) == 0) :
            return 
        if grad_input[0] == None or grad_output[0] == None :
            return 
        if (torch.isnan(grad_input[0]).any() or
            torch.isnan(grad_output[0]).any()) :

            print("\n")
            raise RuntimeError(f"Found NAN in gradient")
    return hook


def get_pts_for_plot(query_idx,eval_seq,tfs,pose) :

    kitti_dir = WORK_PATH+"/datas/datasets/"
    fname = kitti_dir + 'sequences/'+eval_seq+'/velodyne/'+'%06d' % query_idx + '.bin'
    #load points
    xyz = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    # every possible positions
    x, z, y = pose[:,0], pose[:,1], pose[:,2]

    # rotation 1
    out = np.zeros((len(xyz), 3))
    mat = tfs[query_idx][:3,:3]
    for i in range (len(xyz)):
        out[i] = ( mat @ xyz[i][:3] ) 
    xyzr = out

    # translation
    pose_q = pose[query_idx]
    pose_q[[1,2]] =  pose_q[[2,1]]
    xyzrf = xyzr[:, :3] + pose_q

    return xyzrf
    
def print_loader(loader,lab) :
    do_dump_image = False
    lit = iter(loader)

    eval_seq = cfg['DATA_CONFIG']['SEQ']
    kitti_dir = WORK_PATH+"/datas/datasets/"
    print("")
    print("=========  loader " + lab + " ===========")
    print("ln : " + str(len(loader)))
    acc = 0
    os.makedirs("plot_" + lab, exist_ok=True)
    sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
    #tfs, pose = load_poses_from_txt(sequence_path + 'poses.txt')
    for ii in lit :
        print("id:" + str(ii['id']) + " gt:" + str(ii['gt']) + " labels:" + str(ii['labels']) ) #+ " gps_label:" + str(ii['gps']))
        acc = acc+1

        if do_dump_image : 
            xyzrf = get_pts_for_plot(int(ii['id']),eval_seq,tfs,pose)
            if int(ii['gt']) > 0:
                xyzrf_gt = get_pts_for_plot(int(ii['gt']),eval_seq,tfs,pose)
            x, z, y = pose[:,0], pose[:,1], pose[:,2]
            plt.figure()
            plt.scatter(xyzrf[:, 0], xyzrf[:, 2], c='b', s=0.05,marker='x')
            if int(ii['gt']) > 0:
                plt.scatter(xyzrf_gt[:, 0], xyzrf_gt[:, 2], c='r', s=0.05,marker='o')
            plt.scatter(x,z,c='g', s=0.1)
            plt.xlabel("X")
            plt.ylabel("z")
            plt.title("query "+str(ii['id']) )
            plt.axis('equal')
            plt.savefig('plot_' + lab + '/query_'+str(ii['id']) +'.png',dpi=600)

        if(acc > 64) :
            print("....")
            break
    print("========= end loader ===========")


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_shape', 'image_pad_shape', 'image_rescale_shape','labels','index','input_ids','transformation_3d_list','transformation_3d_params','transformation_2d_list','transformation_2d_params','batch_size','gt','gts','id','gps','hilbert', 'id_pcd_positif','id_pcd_negatif','other_id_pcd_negatif',  'lidar_values', 'lidar_values_load']: #ajout key truth
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        
        if not str(m).startswith("Linear8bitLt") :
            m.reset_parameters()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--model_name', type=str, default="git", help='checkpoint to start from')
    parser.add_argument('--use_sop', type=str, default="True", help='')    
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training GD-MAE')
    parser.add_argument('--eval_steps', type=int, default=100, required=False, help='batch size for training DSI')
    parser.add_argument('--evaluation_strategy', type=str, default="steps", required=False, help='batch size for training DSI')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False, help='batch size for training DSI')
    parser.add_argument('--warmup_steps', type=int, default=1000, required=False, help='batch size for training DSI')
    parser.add_argument('--save_steps', type=int, default=0, required=False, help='batch size for training DSI')
    parser.add_argument('--logging_steps', type=int, default=1, required=False, help='batch size for training DSI')
    parser.add_argument('--train_batch_size', type=int, default=32, required=False, help='batch size for training DSI')
    parser.add_argument('--git_checkpoint', type=str, default=None, help='specify the config for training')
    parser.add_argument('--eval_checkpoint', type=str, default=None, help='specify the config for training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='specify the config for training')

    
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, required=False, help='batch size for training DSI')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, required=False, help='batch size for training DSI')


    parser.add_argument('--do_train', type=str, default="False", help='')
    parser.add_argument('--do_eval', type=str, default="False", help='')
    parser.add_argument('--do_eval_partial', type=str, default="False", help='')
    parser.add_argument('--do_preprocess', type=str, default="False", help='')
    parser.add_argument('--do_dump_dict_gt', type=str, default="False", help='')
    
    parser.add_argument('--adam_epsilon', type=float, default=1e-05, required=False, help='adam_epsilon')
    parser.add_argument('--dataset_train_len', type=int, default=64, required=False, help='adam_epsilon')
    parser.add_argument('--dataset_eval_len', type=int, default=16, required=False, help='adam_epsilon')
    parser.add_argument('--learning_rate', type=float, default=1e-07, required=False, help='adam_epsilon')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--dispatch_batches', type=bool, default=True, required=False, help='')
    
    parser.add_argument('--reset_model', type=bool, default=False, required=False, help='')
    parser.add_argument('--weighted_crossentropy', type=bool, default=False, required=False, help='')   
    parser.add_argument('--adam_beta1', type=float, default=0.9, required=False, help='adam_epsilon')
    parser.add_argument('--adam_beta2', type=float, default=0.999, required=False, help='adam_epsilon')
    parser.add_argument('--num_train_epochs', type=int, default=3, required=False, help='adam_epsilon')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', type=int, default=-1, help='seed')    
    #parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=1, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=10, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--remove_unused_columns', type=bool, default=False, help='')

    
    parser.add_argument('--dataloader_pin_memory', type=bool, default=False, help='')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='')
    parser.add_argument('--output_dir', type=str, default=None, help='output_dir')
    parser.add_argument('--log3dnet_dir', type=str, default=None, help='log3dnet')

    # added
    parser.add_argument('--save_hit_file',  type=str, default='hit.txt', help='file with hit score')
    parser.add_argument('--id_max_length', type=int, default=10, required=False, help='adam_epsilon')
    parser.add_argument('--eval_chkt', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", required=False, help='LR scheduler type')
    parser.add_argument('--num_cycles', type=int, default=1, required=False, help='restart cycles')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.sync_bn = args.sync_bn or cfg.OPTIMIZATION.get('SYNC_BN', False)
    

    return args, cfg


##############################################################################
# Optimisé make_compute_metrics
# #############################################################################
def make_compute_metrics(tokenizer, logger, rank, positions_database, label_mapping, label_mapping_val, save_file_name):
#def make_compute_metrics(tokenizer, logger, rank, train_set, sequence_path, d=None):
    def compute_metrics(eval_preds):
        hit_at_1, hit_at_10 = 0, 0
        print("inside compute metrics")
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            
            #query_id = label_mapping.get(label_id, label_id) # (keyname, value=value to return if the specified key does not exist)
            query_id = label_mapping_val[label_id]
            answers_ids = [label_mapping.get(x, x) for x in rank_list]
            #print("query_id ", query_id)
            #print("answers_ids ", answers_ids)
            # Position-based metrics
            #label_id_gps = positions_database[int(query_id)]
            #rank_list_gps = [positions_database[int(x)] for x in answers_ids]

            label_id_gps = positions_database.get(query_id, [-10, -10])
            rank_list_gps = [positions_database.get(x, [-10, -10]) for x in answers_ids]

            print("query ", os.path.basename(query_id) )
            print("Top1  ", os.path.basename(answers_ids[0]) )
            
            rank_list_dist = [
                math.dist(label_id_gps[:2], rank_list_gps[i][:2]) for i in range(len(rank_list_gps))
            ]
            #print("rank_list_dist ", rank_list_dist)
            
            rank_list_dist_filter = [1 if dist <= 1 else 0 for dist in rank_list_dist]
            print("rank_list_dist_filter ", rank_list_dist_filter)
            
            hits_clos = np.where(np.array(rank_list_dist_filter)[:10] == 1)[0]
            if hits_clos.size > 0:
                hit_at_10 += 1
                if hits_clos[0] == 0: 
                    hit_at_1 += 1
                    
        #hit_at_1_tensor = torch.tensor(hit_at_1, device="cuda")
        #hit_at_10_tensor = torch.tensor(hit_at_10, device="cuda")
        #dist.all_reduce(hit_at_1_tensor, op=dist.ReduceOp.SUM)
        #dist.all_reduce(hit_at_10_tensor, op=dist.ReduceOp.SUM)

        total_predictions = len(eval_preds.predictions)

        #######################################################################
        # save metrics
        #######################################################################
        with open(save_file_name, 'a') as f:
            f.write(str(hit_at_1 / total_predictions ) + " " + str(hit_at_10 / total_predictions ) + "\n")
        #f.close()
        print("hit_at_1 ", hit_at_1, " hit_at_10 ", hit_at_10, " total_predictions " , total_predictions )

        #######################################################################
        #######################################################################
        
        return {
            "Hits@1": hit_at_1 / total_predictions,
            "Hits@10": hit_at_10 / total_predictions,
        }
    
    return compute_metrics

##############################################################################
# Optimisé DSITrainer
# #############################################################################
class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, LIK, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        print(" id_max_length ",  id_max_length)
      
        self.id_max_length = id_max_length
        self.LIK = LIK
        self.per_device_train_batch_size = kwds['args'].per_device_train_batch_size
        self.per_device_eval_batch_size = kwds['args'].per_device_eval_batch_size

    def compute_loss(self, model, inputs, return_outputs=False): # 1
        del inputs['ids']
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss 

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        model.eval()
        
        vv = self.tokenizer.batch_decode(inputs["labels"],skip_special_tokens=True)
        self.ll1 = []

        with torch.no_grad():
            # Beam search parameters
            batch_size = inputs['pixel_values'].size(0)
            nb_beam = self.id_max_length
            inputs['lidar_values']['batch_size'] = self.per_device_eval_batch_size
            
            # Remove ids from inputs
            ids = inputs.pop('ids')
            
            self.id_max_length = 10
            nb_beam = 10
            
            batch_beams_dict = model.generate(
                pixel_values=inputs['pixel_values'],
                lidar_values=inputs['lidar_values'],
                points=None,
                max_length= self.id_max_length, #8
                num_beams=nb_beam, #8
                num_return_sequences=nb_beam, #8
                eos_token_id=self.tokenizer.eos_token_id, #102
                pad_token_id=self.tokenizer.pad_token_id, #0
                bos_token_id=self.tokenizer.bos_token_id, #101
                renormalize_logits=True,
                early_stopping=False, #True,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                return_dict_in_generate=True,                
                output_scores = True,
            )

            
            
            # Extract generated sequences and scores
            batch_beams = batch_beams_dict['sequences']
            #seq_score = batch_beams_dict['sequences_scores'].reshape(batch_size, nb_beam)

            # Pad sequences to the maximum length
            #batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length, self.tokenizer)
            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length, self.tokenizer)
            
            # Reshape beams for batch-wise operations
            batch_beams = batch_beams.reshape(batch_size, nb_beam, -1)


            # Optional: Debugging/logging for predictions
            batch_score =0
            for ii in range(batch_size):
                decoded_labels = self.tokenizer.batch_decode(batch_beams[ii].cpu(), skip_special_tokens=True)
                print(f"IDs: {ids[ii]}")
                print(f"Labels: {self.tokenizer.decode(inputs['labels'][ii], skip_special_tokens=True)}")
                print(f"Beams: {decoded_labels}")
                #print(f"Scores: {seq_score[ii]}")
                print("----")

                if decoded_labels == self.tokenizer.decode(inputs['labels'][ii], skip_special_tokens=True):
                    batch_score += 1
            print("% correct ", batch_score/batch_size)

               
        return None, batch_beams, inputs['labels'] # loss, logits, labels


    def _pad_tensors_to_max_len(self, tensor, max_length, tokenizer):
        """
        Pads tensor to a specified maximum length using the pad token ID.
        """
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        tensor[tensor == -100] = pad_token_id  # Replace masked tokens
        padded_tensor = pad_token_id * torch.ones(
            (tensor.size(0), max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, :tensor.size(1)] = tensor
        return padded_tensor

@dataclass

##############################################################################
# Optimisé IndexingCollator
# #############################################################################
class IndexingCollator(DataCollatorWithPadding):
    def __init__(self, label_tokenizer, padding, id_max_length, processor, batch_size):
        super().__init__(label_tokenizer, padding)
        self.processor = processor
        self.batch_size = batch_size
        self.id_max_length = id_max_length
        self.tokenizer = label_tokenizer
        
    def __call__(self, features):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Extract features
    
        input_ids = torch.vstack([x['input_ids'] for x in features])
        labels = input_ids.clone()
        ids = [x['id'] for x in features]
        attention_mask = torch.vstack([x['attention_mask'] for x in features])
        pixel_values = torch.cat([x['pixel_values'] for x in features], dim=0).to(device=device)

        # Process `attention_mask` and `input_ids`
        attention_mask[input_ids == self.tokenizer.eos_token_id] = 0
        input_ids[input_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id

        # Prepare `inputs` dictionary
        inputs = {
            'input_ids': input_ids.to(device=device),
            'labels': labels.to(device=device),
            'ids': ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask.to(device=device),
        }
        
        # Process LIDAR values
        lidar_values = self._prepare_lidar_values(features, device)
        inputs['lidar_values'] = lidar_values
        inputs['lidar_values']['pixel_values'] = inputs['pixel_values']
        
        # Load LIDAR data to GPU if available
        if device == "cuda":
            load_data_to_gpu(inputs['lidar_values'])

        return inputs

    def _prepare_lidar_values(self, features, device):
        lidar_val = {'batch_size': self.batch_size}
        
        feature_dict = {k: [x[k] for x in features] for k in features[0].keys()}
        
        for key, val in feature_dict.items():
            if key in ['frame_id', 'id_pcd_positif', 'id_pcd_negatif', 'other_id_pcd_negatif']:
                lidar_val[key] = np.stack(val, axis=0) 
                
            if key in ['frame_id_desc', 'id_pcd_positif_desc', 'id_pcd_negatif_desc', 'other_id_pcd_negatif_desc']:
                lidar_val[key] = val 
                
        return lidar_val
        ##############################################################################
        ##############################################################################
        
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_name="", threshold=1):
        # metric_name can be any key returned in the evaluation logs (e.g., 'eval_loss', 'eval_accuracy')
        self.metric_name = metric_name
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Check if the evaluation metric is higher than the threshold
        eval_metric = metrics.get(self.metric_name)
        if eval_metric and eval_metric > self.threshold:
            print(f"Stopping training early! {self.metric_name} = {eval_metric}")
            control.should_training_stop = True



def Print_active_layers_git(args, logger, model_dsi, model_name):
    grad_module_name = []
    if model_name == "git" :
        for name, param in model_dsi.named_parameters():
            if (
                name.startswith("vision_model") or
                name.startswith("language_model") or
                name.startswith("git.image_encoder") or
                name.startswith("git.visual_projection") or
                name.startswith("git.embeddings.word_embeddings.weight") or
                name.startswith("git.embeddings.position_embeddings.weight")                    
            ) :
                param.requires_grad = False
            if (
                name.startswith("VOID") 
            ) :
                param.requires_grad = True
            if param.requires_grad == True  :
                grad_module_name.append(name)
                if args.local_rank == 0 :
                    logger.info(name + "\t =>" + str(param.requires_grad))
    return



def main():
    ### ===== START ===========
    # parametres
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args, cfg = parse_config()
    ID_MAX_LENGTH = args.id_max_length 
    MAX_LENGTH = ID_MAX_LENGTH

    model_name = args.model_name
    dataset_train_len = args.dataset_train_len
    
    dataset_eval_len = args.dataset_eval_len

    checkp_to_eval = args.eval_chkt
    
    do_overfit = True
    random_seed = int(args.fix_random_seed)
    do_use_sop = eval(args.use_sop)
    if args.launcher == "pytorch" : 
        args.local_rank = int(os.environ['LOCAL_RANK'])


    sequence_path = cfg['DATA_CONFIG']['DATA_PATH'] + cfg['DATA_CONFIG']['SEQ']
    
    ##########################################################################################
    save_file_name = args.save_hit_file # new for naming file hit score
    with open(save_file_name, 'w') as f:
        print('create and reset the hit txt output file ', save_file_name)
    f.close()
    
    with open("loss.txt", 'w') as f:
        print('create and reset the loss txt output file loss.txt')
    f.close()

    with open("distribution.txt", 'w') as f:
        print('create and reset the loss txt output file loss.txt')
    f.close()

    ##########################################################################################

    do_train = eval(args.do_train)
    do_eval = eval(args.do_eval)
    do_eval_partial = eval(args.do_eval_partial)
    do_preprocess = eval(args.do_preprocess)
    do_dump_dict_gt = eval(args.do_dump_dict_gt)

    print("============================================")
    print("do_eval:"  + str(do_eval))
    print("do_train:"  + str(do_eval))
    print("do_eval_partial:"  + str(do_eval_partial))
    print("do_preprocess: "  + str(do_preprocess))
    print("do_dump_dict_gt: "  + str(do_dump_dict_gt))
    print("============================================")

    ### ==== ARGUMENT PARSER  =====
    ## T5 Args parser 
    parser = HfArgumentParser((TrainingArguments,))
  
    ## GD-MAE parser
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if random_seed > 0 :
        common_utils.set_random_seed(random_seed)
    print("RANDOM SEED:" + str(random_seed))
    training_args, remaiening = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    print("training_args.lr_scheduler_type ", training_args.lr_scheduler_type)
    print("training_args.warmup_ratio:", training_args.warmup_ratio)

    batch_size = training_args.per_device_train_batch_size
    ori_train_batch_size  = training_args.per_device_train_batch_size
    ori_eval_batch_size  = training_args.per_device_eval_batch_size
    args.batch_size = batch_size

    
    ### ===== LOGER ==========        
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (output_dir / 'sop').mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    if args.local_rank == 0 :
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        # -->
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
        if dist_train:
            logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)
        # -->
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    ### ===== DATALOADER ===========
    # -----------------------create dataloader & network & optimizer---------------------------

    def initialize_dataloader(cfg, args, logger, training=True):
        dataset, loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=(args.launcher != 'none'),
            workers=args.workers,
            logger=logger,
            training=training,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=args.epochs,
        )
        return dataset, loader, sampler

    if args.local_rank == 0 :
        logger.info("Initializing dataset and dataloader...")
        
    root_path1 = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v3"
    root_path2 = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/lidarhd_v2"
    
    cfg['DATA_CONFIG']['DATA_PATH'] = root_path1
    #train_set, train_loader, _ = initialize_dataloader(cfg, args, logger, training=True)
    eval_set, eval_loader, _ = initialize_dataloader(cfg, args, logger, training=False)


    cfg['DATA_CONFIG']['DATA_PATH'] = root_path2
    eval_set2, eval_loader2, _ = initialize_dataloader(cfg, args, logger, training=False)
    
    
    # Determine subset lengths
    """
    train_len = (
    int(args.dataset_train_len)
    if args.dataset_train_len > 0
    else len(train_set)
    )
    """
    
    eval_len = (int(args.dataset_eval_len) if args.dataset_eval_len > 0 else len(eval_set))
    
    eval_len2 = (int(args.dataset_eval_len) if args.dataset_eval_len > 0 else len(eval_set2))
    
    # Create subsets
    #train_subset = torch.utils.data.Subset(train_set, range(0, train_len))
    eval_subset = torch.utils.data.Subset(eval_set, range(0, eval_len))
    eval_subset2 = torch.utils.data.Subset(eval_set2, range(0, eval_len2))
    
    # Log dataset information
    #print_loader(train_subset, 'train')
    print_loader(eval_subset, 'eval')
    print_loader(eval_subset2, 'eval2')


    ### ========= build Models ===========
    work_path = os.getenv('WORKSF')
    
    model_paths = {
        "git_base": os.path.join(work_path, "datas/transformers/git-base-lhd"),
        "git_large": os.path.join(work_path, "datas/transformers/git-large-coco"),
        "blip2": os.path.join(work_path, "datas/transformers/blip2-opt-2.7b"),
        "bert_base": os.path.join(work_path, "datas/transformers/bert-base-uncased"),
    }
    if args.local_rank == 0 :
        logger.info(f"Model paths: {model_paths}")


    ##############################################################################
    ### === Blig2 / GIT ===
    ##############################################################################
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dsi = None
    tokenizer = None
    
    if model_name == "git" or args.git_checkpoint is not None:
        model_dsi_path = args.git_checkpoint if args.git_checkpoint else model_paths["git_base"]
        if args.local_rank == 0 :
            logger.info(f"Initializing GIT model from {model_dsi_path}...")
        
        # Load GIT configuration and model
        config = AutoConfig.from_pretrained(model_dsi_path)

        model_dsi = GitForCausalLM(config).to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_dsi_path)
        
        # Set up lidar model and optionally restore weights
        model = None
        
        model_dsi.set_lidar_model(model, SOP(signed_sqrt=False, do_fc=False), do_use_sop, eval_set.root_path)

        """
        (Pdb) SOP(signed_sqrt=False, do_fc=False)
        SOP(
          (fc1): Linear(in_features=256, out_features=128, bias=True)
          (fc2): Linear(in_features=128, out_features=64, bias=True)
        )
        (Pdb) do_use_sop,
        (True,)
        """
        if args.git_checkpoint:
            if args.local_rank == 0 :
                logger.info("Restoring GIT input/output embeddings and lidar parameters...")
            input_embeddings = copy.deepcopy(model_dsi.git.get_input_embeddings())
            output_embeddings = copy.deepcopy(model_dsi.get_output_embeddings())
            bt_norm = copy.deepcopy(model_dsi.lidar_encoder.bt_norm)
            lidar_projection = copy.deepcopy(model_dsi.lidar_encoder.lidar_projection)
            model_dsi.set_input_embeddings(input_embeddings)
            model_dsi.set_output_embeddings(output_embeddings)
            model_dsi.lidar_model.set_lidar_encoder(model, lidar_projection, bt_norm)
                

    
    elif model_name == "blip2":
        model_dsi_path = model_paths["blip2"]
        if args.local_rank == 0 :
            logger.info(f"Initializing BLIP2 model from {model_dsi_path}...")
    
        # Load BLIP2 configuration and model
        config = AutoConfig.from_pretrained(model_dsi_path)
        model_dsi = Blip2ModelQuerryLearning(config=config).to(device=device).type(torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_paths["bert_base"])
        
        # Reset BLIP2 parameters if specified
        if args.reset_model:
            if args.local_rank == 0 :
                logger.info("Resetting BLIP2 Q-former and weights...")
            model_dsi.reset_q()
            try:
                model_dsi.qformer.apply(weight_reset)
            except Exception as e:
                logger.error(f"Failed to reset Q-former weights: {traceback.format_exc()}")
        
        # Set lidar model with SOP
        model_sop = SOP(signed_sqrt=False, do_fc=False)
        model_dsi.lidar_model.sop = model_sop
        if args.git_checkpoint:
            if args.local_rank == 0 :
                logger.info("Restoring BLIP2 input/output embeddings and lidar parameters...")
            model_dsi.set_input_embeddings(model_dsi.bert.embeddings, input_embeddings)
            model_dsi.set_output_embeddings(output_embeddings)
            model_dsi.lidar_model.set_lidar_encoder(model, lidar_projection, bt_norm)
        else:
            model_dsi.lidar_model.set_lidar_model(model, model_sop, do_use_sop, eval_set.root_path)
    
    else:
        logger.error(f"Unsupported model name: {model_name}. Must be 'git' or 'blip2'.")


        
    ##############################################################################
    ### === Vocabulary ===
    ##############################################################################

    ### ===== Processor / Tokenizer =====
    processor = AutoProcessor.from_pretrained(model_dsi_path)
    spe_tok = ['[CLS]', '[MASK]', '[PAD]', '[SEP]','[BOS]','[EOS]']
    ukn = tokenizer.convert_tokens_to_ids('[UNK]') # = 100
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token) # ' '
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) # ' '
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) # = 0
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token) # = 102
    tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token) # = 100
    empt_tk = tokenizer('') # {'input_ids': [101, 102], 'attention_mask': [1, 1]}
    if len(empt_tk.input_ids) == 2 :
        tokenizer.bos_token_id = empt_tk.input_ids[0] # = 101
        tokenizer.eos_token_id = empt_tk.input_ids[1] # = 102
    model_dsi.set_tokenizer(tokenizer,ID_MAX_LENGTH)
    

    ## ==== Vocabulary Filtering / Preprocessing ==== 
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    INT_TOKEN_STR = []
    bad_tk = ['₁','₂','₃','₄','₅','₆','₇','₈','₉','₀','²','¹','³','⁷','⁹','⁰','⁴','⁵','⁶','⁸']
    for token, id in tokenizer.get_vocab().items():
        if token[0] == "#":
            if token[2:].isdigit() and (token[2:] not in bad_tk) :
                INT_TOKEN_IDS.append(id)
                INT_TOKEN_STR.append(token)
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit() and (token[1:] not in bad_tk) :
                INT_TOKEN_IDS.append(id)
                INT_TOKEN_STR.append(token)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
            INT_TOKEN_STR.append(token)
        elif token.isdigit() and (token not in bad_tk) :
            INT_TOKEN_IDS.append(id)
            INT_TOKEN_STR.append(token)
    #INT_TOKEN_IDS.append(tokenizer.bos_token_id)            
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)
    INT_TOKEN_IDS.append(tokenizer.pad_token_id) 
    
    model_dsi.set_vocab(INT_TOKEN_IDS) 
    ############################################################
    # create ID and token lists
    n_subset = [int(x) for x in range(len(eval_subset))] 
    n_set = [int(x) for x in range(len(eval_set))] 
    lid = []
    LIK = []
    for ii in n_subset : lid.append(eval_set.get_label(ii))    
    for ii in lid : LIK.append(tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH).input_ids)

    def build_prefix_dict_filter(LIK):
        prefix_dict = {}
        skip_eval_set = 0
        for seq in LIK: # len trainset
            if skip_eval_set % 5 == 0:
                skip_eval_set += 1
                continue
            skip_eval_set += 1
            for sz in range(len(seq) - 1): # length tokens
                prefix = tuple(seq[:sz])  
                next_token = seq[sz]  # The next token
                
                if prefix in prefix_dict:
                    prefix_dict[prefix].add(next_token) 
                else:
                    prefix_dict[prefix] = {next_token}  
        return {k: list(v) for k, v in prefix_dict.items()}  # Convert sets to lists

    
    n_subset = [int(x) for x in range(len(eval_subset))] 
    lid = [eval_set.get_label(ii) for ii in n_subset]
    LIK = [tokenizer(ii, padding="max_length", max_length=ID_MAX_LENGTH).input_ids for ii in lid]
    prefix_dict = build_prefix_dict_filter(LIK) # bien pour toutes les zones
    
    # Optimized restrict_decode_vocab
    def restrict_decode_vocab_v3(batch_idx, prefix_beam):
        pfb = tuple(prefix_beam.cpu().numpy())  
        return prefix_dict.get(pfb, [102])
    
    #update object
    """
    train_set.tokenizer = tokenizer
    train_set.image_processor = processor
    train_set.ID_MAX_LENGTH = ID_MAX_LENGTH
    """    
    eval_set.tokenizer = tokenizer
    eval_set.image_processor = processor
    eval_set.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set2.tokenizer = tokenizer
    eval_set2.image_processor = processor
    eval_set2.ID_MAX_LENGTH = ID_MAX_LENGTH

    
    # enter class indexing collator
    data_collator=IndexingCollator(
        tokenizer,
        padding='longest',
        processor=processor,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 

    logger.info("============== FULL NETWORK STATE =================")
    for name, param in model_dsi.named_parameters() : logger.info(name + "\t =>" + str(param.requires_grad)) 
    logger.info("============== FREE NETWORK STATE =================")

    ### ====== Print active layers git =========
    Print_active_layers_git(args, logger, model_dsi, model_name)  
    logger.info("============== NETWORK STATE =================")

    ### ====== checkpoint =========
    work_path = os.getenv('WORK')
    CHECK_ROOT= work_path + "/checkpoints/"
    checkpoint_dir = CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag

    if not os.path.isdir(checkpoint_dir) :
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)


    ################################################################################################################################################
    # Load experts training dataset
    ################################################################################################################################################


    
    def load_set_ids(train_indices_path, root_path=root_path1):
        print("indices_path ", train_indices_path)
        f = open(root_path +"/"+ train_indices_path) 
        train_indices = json.load(f)
        f.close()
        return train_indices
        
    def load_training_set_ids(train_indices_path, val_indices_path):
        print("train_indices_path ", train_indices_path)
        print("val_indices_path ", val_indices_path)
        f = open(sequence_path +"/"+ train_indices_path) 
        train_indices = json.load(f)
        f.close()
        
        
        return train_indices

    
    # LOAD OUEST
    train_indicesa = load_set_ids("zone_A_dsi_train_list.json")
    train_indicesb = load_set_ids("zone_B_dsi_train_list.json")
    train_indicesc = load_set_ids("zone_C_dsi_train_list.json")
    train_indicesd = load_set_ids("zone_D_dsi_train_list.json")
    train_indicese = load_set_ids("zone_E_dsi_train_list.json")
    
    data = {"A1": train_indicesa, "B1": train_indicesb, "C1": train_indicesc, "D1": train_indicesd, "E1": train_indicese}


    val_indicesa = load_set_ids("zone_A_dsi_val_list.json")
    val_indicesb = load_set_ids("zone_B_dsi_val_list.json")
    val_indicesc = load_set_ids("zone_C_dsi_val_list.json")
    val_indicesd = load_set_ids("zone_D_dsi_val_list.json")
    val_indicese = load_set_ids("zone_E_dsi_val_list.json")

    data_val = {"A1": val_indicesa, "B1": val_indicesb, "C1": val_indicesc, "D1": val_indicesd, "E1": val_indicese}

    
    eval_indicesa = load_set_ids("zone_A_dsi_eval_list.json")
    eval_indicesb = load_set_ids("zone_B_dsi_eval_list.json")
    eval_indicesc = load_set_ids("zone_C_dsi_eval_list.json")
    eval_indicesd = load_set_ids("zone_D_dsi_eval_list.json")
    eval_indicese = load_set_ids("zone_E_dsi_eval_list.json")

    data_eval = {"A1": eval_indicesa, "B1": eval_indicesb, "C1": eval_indicesc, "D1": eval_indicesd, "E1": eval_indicese}

    
    print("train len ", len(train_indicesa + train_indicesb + train_indicesc + train_indicesd + train_indicese ) )
    print("val len ", len(val_indicesa + val_indicesb + val_indicesc + val_indicesd + val_indicese) )
    print("eval len ", len(eval_indicesa + eval_indicesb + eval_indicesc + eval_indicesd + eval_indicese))


    # LOAD EST
    train_indicesa2 = load_set_ids("zone_A_dsi_train_list.json",root_path2)
    train_indicesb2 = load_set_ids("zone_B_dsi_train_list.json",root_path2)
    train_indicesc2 = load_set_ids("zone_C_dsi_train_list.json",root_path2)
    train_indicesd2 = load_set_ids("zone_D_dsi_train_list.json",root_path2)
    
    data2 = {"A0": train_indicesa2, "B0": train_indicesb2, "C0": train_indicesc2, "D0": train_indicesd2}

    val_indicesa2 = load_set_ids("zone_A_dsi_val_list.json",root_path2)
    val_indicesb2 = load_set_ids("zone_B_dsi_val_list.json",root_path2)
    val_indicesc2 = load_set_ids("zone_C_dsi_val_list.json",root_path2)
    val_indicesd2 = load_set_ids("zone_D_dsi_val_list.json",root_path2)

    data_val2 = {"A1": val_indicesa2, "B1": val_indicesb2, "C1": val_indicesc2, "D1": val_indicesd2}

    
    eval_indicesa2 = load_set_ids("zone_A_dsi_eval_list.json",root_path2)
    eval_indicesb2 = load_set_ids("zone_B_dsi_eval_list.json",root_path2)
    eval_indicesc2 = load_set_ids("zone_C_dsi_eval_list.json",root_path2)
    eval_indicesd2 = load_set_ids("zone_D_dsi_eval_list.json",root_path2)

    data_eval2 = {"A0": eval_indicesa2, "B0": eval_indicesb2, "C0": eval_indicesc2, "D0": eval_indicesd2}
    data_eval = {"A1": eval_indicesa, "B1": eval_indicesb, "C1": eval_indicesc, "D1": eval_indicesd, "E1": eval_indicese,"A0": eval_indicesa2, "B0": eval_indicesb2, "C0": eval_indicesc2, "D0": eval_indicesd2}
    
    print("train len ", len(train_indicesa2 + train_indicesb2 + train_indicesc2 + train_indicesd2  ) )
    print("val len ", len(val_indicesa2 + val_indicesb2 + val_indicesc2 + val_indicesd2 ) )
    print("eval len ", len(eval_indicesa2 + eval_indicesb2 + eval_indicesc2 + eval_indicesd2 ))


    ########################################################
    # ground truth frame_id distribution
    ########################################################
    path_to_ids = {}

    all_names = [
        train_indicesa,
        train_indicesb,
        train_indicesc,
        train_indicesd,
        train_indicese,
        train_indicesa2,
        train_indicesb2,
        train_indicesc2,
        train_indicesd2,
    ]

    
    for idx, name_list in enumerate(all_names):
        for path in name_list:
            base = os.path.basename(path)[:-4] + '.pt'  # extract filename
            if base not in path_to_ids:
                path_to_ids[base] = []
            path_to_ids[base].append(idx)



    
    all_names = [
        val_indicesa,
        val_indicesb,
        val_indicesc,
        val_indicesd,
        val_indicese,
        val_indicesa2,
        val_indicesb2,
        val_indicesc2,
        val_indicesd2,
    ]

    
    for idx, name_list in enumerate(all_names):
        for path in name_list:
            base = os.path.basename(path)[:-4] + '.pt'   # extract filename
            if base not in path_to_ids:
                path_to_ids[base] = []
            path_to_ids[base].append(idx)


    all_names = [
        eval_indicesa,
        eval_indicesb,
        eval_indicesc,
        eval_indicesd,
        eval_indicese,
        eval_indicesa2,
        eval_indicesb2,
        eval_indicesc2,
        eval_indicesd2,
    ]
    
    for idx, name_list in enumerate(all_names):
        for path in name_list:
            base = os.path.basename(path)[:-4] + '.pt'   # extract filename
            if base not in path_to_ids:
                path_to_ids[base] = []
            path_to_ids[base].append(idx)

    print(len(path_to_ids))
    
    expert_labels = path_to_ids
    
    
    #import pdb; pdb.set_trace() # expert_labels['LHD_FXX_0656_6860_PTS_O_LAMB93_IGN69.copc_17_10_47.bin']
    ########################################################

    
    ################################################################################################################################################
    # Mixture-of-Experts Setup
    ################################################################################################################################################
    
    root_path =  cfg['DATA_CONFIG']['DATA_PATH']

    def load_tensor(filename,lab) :
        xx = []
        for ll in filename :
            fname = cfg['DATA_CONFIG']['DATA_PATH'] + "/" +  lab + "/" + (os.path.splitext(ll)[0]+'.pt')
            if os.path.isfile(fname) : 
                xx1 = torch.load(fname)
                xx.append(xx1)
            else :
                print("TENSOR NOT FOUND")
                print(fname)
                return None
        return torch.stack(xx)

    def load_tensor_just_one(nn,lab) :
        xx = []
        fname = root_path + "/" +  lab + "/" +  (nn+'.pt')
        if os.path.isfile(fname) : 
            xx1 = torch.load(fname, map_location='cuda')
            xx.append(xx1)
        else :
            print("TENSOR NOT FOUND")
            print(fname)
            return None
        return torch.stack(xx)


    checkpoint_paths = [
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_A1/checkpoint-4100/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_B1/checkpoint-4200/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_C1/checkpoint-4200/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_D1_v2/checkpoint-4300/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_E1/checkpoint-4100/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_new_tokenizer_A0/checkpoint-6300/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_B0/checkpoint-7200/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_C02/checkpoint-7000/pytorch_model.bin",
                    "/lustre/fswork/projects/rech/dki/ujo91el/checkpoints/git_hilbert_train_DO2/checkpoint-1400/pytorch_model.bin"
                   ]
    

    transformer_list = []

    import copy
    transformer_list = []
    for chkp in checkpoint_paths:
        model_copy = copy.deepcopy(model_dsi)          # clone current architecture
        state_dict = torch.load(chkp, map_location="cpu")
        model_copy.load_state_dict(state_dict, strict=False)
        model_copy.eval()
        model_copy.to(device)
        transformer_list.append(model_copy)


    
    from transformers.modeling_outputs import ModelOutput

    @dataclass
    class MoEOutput(ModelOutput):
        loss: torch.FloatTensor = None
        logits: torch.FloatTensor = None
        gate_weights: torch.FloatTensor = None


    # the below code is almost all transcribed from the official tensorflow version, from which the papers are written
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py
    
    # gating network

    def top1(t):
        values, index = t.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def cumsum_exclusive(t, dim=-1):
        num_dims = len(t.shape)
        num_pad_dims = - dim - 1
        pre_padding = (0, 0) * num_pad_dims
        pre_slice   = (slice(None),) * num_pad_dims
        padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
        return padded_t[(..., slice(None, -1), *pre_slice)]


    # pytorch one hot throws an error if there are out of bound indices.
    # tensorflow, in contrast, does not throw an error
    def safe_one_hot(indexes, max_length):
        max_index = indexes.max() + 1
        return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

    
    import torch.nn.functional as F
    MIN_EXPERT_CAPACITY = 4


    class ExpertClassifier(nn.Module):
        def __init__(self, input_dim=256, num_experts=5):
            super(ExpertClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
    
                nn.Linear(256, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
    
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
    
                nn.Linear(256, num_experts)
            )
        
        def forward(self, x):
            return self.model(x)

    
    class Top2Gating(nn.Module):
        def __init__(
            self,
            dim,
            num_gates,
            eps = 1e-9,
            outer_expert_dims = tuple(),
            second_policy_train = 'random',
            second_policy_eval = 'random',
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.):
            super().__init__()
    
            self.eps = eps
            self.num_gates = num_gates
            
            self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
            
            self.gate = ExpertClassifier(input_dim=256, num_experts=9)

            self.gate.load_state_dict(torch.load("expert_gate_est_ouest.pth"))
            
            
            self.gate.to(next(self.parameters()).device)
            self.gate.eval()
            
            
            self.second_policy_train = second_policy_train
            self.second_policy_eval = second_policy_eval
            self.second_threshold_train = second_threshold_train
            self.second_threshold_eval = second_threshold_eval
            self.capacity_factor_train = capacity_factor_train
            self.capacity_factor_eval = capacity_factor_eval
    
        def forward(self, x, importance = None):
            #*_, b, group_size, dim = x.shape
            group_size, dim = x.shape #[1,256]
            #b = group_sizes
            num_gates = self.num_gates
    
            if self.training: # true
                policy = self.second_policy_train # 'random'
                threshold = self.second_threshold_train #0.2
                capacity_factor = self.capacity_factor_train # 1.25
            else:
                policy = self.second_policy_eval
                threshold = self.second_threshold_eval
                capacity_factor = self.capacity_factor_eval
    
            #raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
            #raw_gates = torch.einsum('...d,...de->...e', x, self.w_gating)
            #raw_gates = raw_gates.softmax(dim=-1) # tensor([[[[0.7157, 0.0576, 0.0691, 0.1576]]]])
            #print("og raw gates", raw_gates)
            
            raw_gates = self.gate(x) 
            raw_gates = raw_gates.softmax(dim=-1) # tensor([[[[0.7157, 0.0576, 0.0691, 0.1576]]]])
            #print("new raw gates", raw_gates)
            
            
            # FIND TOP 2 EXPERTS PER POSITON
            # Find the top expert for each position. shape=[batch, group]

            #print( "raw_gates.shape() " , raw_gates.shape )
            
            gate_1, index_1 = top1(raw_gates) # (tensor([[[0.7157]]]), tensor([[[0]]]))
            mask_1 = F.one_hot(index_1, num_gates).float() # tensor([[[[1., 0., 0., 0.]]]])
            density_1_proxy = raw_gates # tensor([[[[0.7157, 0.0576, 0.0691, 0.1576]]]])
            
            if importance is not None: # True
                equals_one_mask = (importance == 1.).float()
                mask_1 *= equals_one_mask[..., None]
                gate_1 *= equals_one_mask
                density_1_proxy = density_1_proxy * equals_one_mask[..., None]
                del equals_one_mask
                
            gates_without_top_1 = raw_gates * (1. - mask_1)
    
            gate_2, index_2 = top1(gates_without_top_1) # (tensor([[[0.1576]]]), tensor([[[3]]]))
            mask_2 = F.one_hot(index_2, num_gates).float() # tensor([[[[0., 0., 0., 1.]]]])


            if importance is not None:
                greater_zero_mask = (importance > 0.).float()
                mask_2 *= greater_zero_mask[..., None]
                del greater_zero_mask

            # normalize top2 gate scores
            denom = gate_1 + gate_2 + self.eps # tensor([[[0.8733]]]) et self.eps = 1e-09
            gate_1 = gate_1 / denom # tensor([[[0.8195]]])
            gate_2 = gate_2 / denom#  tensor([[[0.1805]]])
            
            # BALANCING LOSSES
            # shape = [batch, experts]
            # We want to equalize the fraction of the batch assigned to each expert
            density_1 = mask_1.mean(dim=-2) # tensor([[[1., 0., 0., 0.]]])
            # Something continuous that is correlated with what we want to equalize.
            density_1_proxy = density_1_proxy.mean(dim=-2) # tensor([[[0.7157, 0.0576, 0.0691, 0.1576]]])
            loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2) # tensor(0.1789) * 16

            # Depending on the policy in the hparams, we may drop out some of the
            # second-place experts.
            if policy == "all":
                pass
            elif policy == "none":
                mask_2 = torch.zeros_like(mask_2)
            elif policy == "threshold":
                mask_2 *= (gate_2 > threshold).float()
            elif policy == "random":
                probs = torch.zeros_like(gate_2).uniform_(0., 1.) # tensor([[[0.3690]]])
                mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
            else:
                raise ValueError(f"Unknown policy {policy}")

            
            Top1 = True
            top_all = False
            if Top1:
                print("Gate top-1")
                CUSTOM_GATE = mask_1
            elif top_all:
                print("Gate top-4")
                CUSTOM_GATE = raw_gates
            else:
                print("Gate top-2")
                CUSTOM_GATE = gate_1.unsqueeze(-1) * mask_1 + gate_2.unsqueeze(-1) * mask_2

            dispatch_tensor, combine_tensor, loss = None, None, loss
            return dispatch_tensor, combine_tensor, loss, CUSTOM_GATE
    

    class MoE_DSI(nn.Module):
        def __init__(self, input_dim, experts, tokenizer, expert_labels):
            super().__init__()
            self.num_experts = len(experts)
            self.experts = nn.ModuleList(experts)
            for p in self.experts.parameters():
                p.requires_grad = False
                
            self.tokenizer = tokenizer
            self.expert_labels = expert_labels

            gating_kwargs = {'second_policy_train': 'random', 'second_policy_eval': 'random', 'second_threshold_train': 0.2, 'second_threshold_eval': 0.2, 'capacity_factor_train': 1.25, 'capacity_factor_eval': 2}
            self.gate = Top2Gating(256, num_gates = 9, **gating_kwargs)

        # for training
        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, lidar_values=None, labels=None, **kwargs):
            # ---- Gate input ----

            print('inside forward')

            ########################################################
            # load descriptor from names
            ########################################################
            x = self._make_gate_input(lidar_values)  # -> [batch, 256]
            ########################################################


            dispatch_tensor, combine_tensor, gate_loss, CUSTOM_GATE = self.gate(x)
            gate_weights = CUSTOM_GATE.to(device=lidar_values['pixel_values'].device)


            ########################################################
            # batch accuracy
            ########################################################
            frame_ids = [f[0] for f in lidar_values['frame_id']]
            
            pred_expert = CUSTOM_GATE.argmax(dim=-1)    
            pred_expert_list = pred_expert.detach().cpu().tolist()
            
            # Check assignment correctness
            for fid, pred in zip(frame_ids, pred_expert_list):
                valid_experts = self.expert_labels.get(fid, [])
                is_correct = pred in valid_experts
                print(f"{fid}: predicted={pred}, valid={valid_experts}, correct={is_correct}")

            correct = sum(pred in self.expert_labels.get(fid, []) 
                for fid, pred in zip(frame_ids, pred_expert_list))
            batch_acc = correct / len(frame_ids)
            print(f"Gate assignment accuracy: {batch_acc:.2%}")
            ########################################################
            
            # ---- Experts forward ----

            batch_size = input_ids.shape[0] if input_ids is not None else lidar_values['pixel_values'].shape[0]
            seq_len = labels.shape[1] + 1 if labels is not None else 11 
            vocab_size = 30522

            all_outputs = []
            
            for e, expert in enumerate(self.experts):
                out = expert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    lidar_values=lidar_values,
                    labels=None, 
                    return_dict=True,
                )

  
                logits = out.logits  # [batch, seq_len, vocab]
                logits_max = logits.argmax(dim=-1)


                if labels is not None:
                    vocab_size = 30522
                    print("Expert ", e)
                    shifted_logits = logits[:, 1:-1, :].contiguous() # torch.Size([9, 30522])
                    shifted_labels = labels[:, 1:].contiguous() # torch.Size([9])
                    
                    print( "labels = ", [ self.tokenizer.decode(x, skip_special_tokens=True) for x in labels ])
                    print( "input_ids = ", [ self.tokenizer.decode(x, skip_special_tokens=True) for x in input_ids ])

                    print( "shifted_labels = ", [ self.tokenizer.decode(x, skip_special_tokens=True) for x in shifted_labels ])
              
                    logits_max2 = logits_max[:, 1:-1].contiguous()
                    print( "logit_id = ", [ self.tokenizer.decode(x, skip_special_tokens=True) for x in logits_max2 ])
                    
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
                    #loss = loss_fct(shifted_logits.view(-1, vocab_size), shifted_labels.view(-1))
                    loss = loss_fct(shifted_logits.view(-1, vocab_size), shifted_labels.view(-1))
                    print("loss expert ", e, " : ", loss)
                    
                all_outputs.append(logits)
            # ---- Mix experts ----

            for i in all_outputs: print( i.shape )
            for i in all_outputs: print( "loss experts ", loss_fct( i[:, 1:-1, :].contiguous().view(-1, vocab_size) , shifted_labels.view(-1)) )
            
            all_outputs = torch.stack(all_outputs, dim=1)

            gw = gate_weights.unsqueeze(-1).unsqueeze(-1) 
        
            sum_logits_weighted = torch.sum(all_outputs * gw, dim=1)
            print("loss 1", loss_fct( sum_logits_weighted[:, 1:-1, :].contiguous().view(-1, vocab_size) , shifted_labels.view(-1)) ) # = tensor(9.6808, device='cuda:0', grad_fn=<NllLossBackward0>)
         
            
            probs = torch.softmax(all_outputs, dim=-1)  # [B, E, T, V]
            mixed_probs = torch.sum(probs * gw, dim=1)
            mixed_logits = torch.log(mixed_probs )
            loss = loss_fct(mixed_logits[:, 1:-1, :].contiguous().view(-1, vocab_size), shifted_labels.view(-1))
            print("loss 2", loss ) # = tensor(9.6808, device='cuda:0', grad_fn=<NllLossBackward0>)


            batch_losses = []
            for b in range(batch_size):
                e = gate_weights[b].argmax().item()
                logits_b = all_outputs[b, e]  # [T, V]
                labels_b = shifted_labels[b]
                loss_b = loss_fct(logits_b[1:-1, :].contiguous().view(-1, vocab_size), labels_b.contiguous().view(-1))
                batch_losses.append(loss_b)
            loss3 = torch.stack(batch_losses).mean()
            print("loss 3", loss3 )  # = tensor(9.6808, device='cuda:0', grad_fn=<MeanBackward0>)
        

            
            #supervised_gate_loss = F.cross_entropy(gate_weights, self.expert_labels.to(gate_preds.device))
            
            loss_coef = 1e-2
            loss_coef2 = 1e-1
            print("loss LM", loss)
            print("loss moe", gate_loss) 
            new_loss = loss + loss_coef * gate_loss
            #new_loss = loss + loss_coef * gate_loss + loss_coef2 *supervised_gate_loss

            print("loss tot", new_loss)

            with open("loss.txt", 'a') as f:
                f.write(str(new_loss) + " " + str(loss) + " " + str(gate_loss) + "\n")
                    
            print(f"Total training steps: {num_training_steps}")
            print(f"Initial LR: {optimizer.param_groups[0]['lr']}")

            return MoEOutput(
                loss=new_loss,
                logits=mixed_logits,
                gate_weights=gate_weights,
            )
        
        def load_tensor(filename,lab) :
            xx = []
            for ll in filename :
                fname = cfg['DATA_CONFIG']['DATA_PATH'] + "/" +  lab + "/" + (os.path.splitext(ll)[0]+'.pt')

                if os.path.isfile(fname) : 
                    xx1 = torch.load(fname)
                    xx.append(xx1)
                else :
                    print("TENSOR NOT FOUND")
                    print(fname)
                    return None
            return torch.stack(xx)

        def _make_gate_input(self, lidar_values):
            # moyen d'opitmiser ici
            filename = lidar_values['frame_id']
            #input_vec = load_tensor_just_one(filename[0], "256_desc_2025-06-23_11-22-13_run_0_4").to(device).float()
            input_vec = lidar_values['frame_id_desc'][0][0]
            return input_vec

        # for validation / evaluation
        def generate(self, x=None, pixel_values=None, lidar_values=None, **kwargs):
            #print('inside generate')
            ########################################################
            # load descriptor from names
            ########################################################
            x = self._make_gate_input(lidar_values)  # -> [batch, 256]
            ########################################################

            dispatch_tensor, combine_tensor, gate_loss, CUSTOM_GATE = self.gate(x)
            gate_weights = CUSTOM_GATE.to(device=lidar_values['pixel_values'].device)
            #gate_weights = torch.tensor([0.5, 0.5, 0, 0]).to(device=lidar_values['pixel_values'].device)
            
            ########################################################
            # predicted expert per sample
            ########################################################
            pred_expert = CUSTOM_GATE.argmax(dim=-1)        
            #print("experts distribution", torch.bincount(pred_expert, minlength=9))
            ########################################################

            all_outputs = []
            all_sequences_scores = []
            scores_shape = None
            num_beams = 10
            batch_size, num_experts = CUSTOM_GATE.shape
            scores_shape = [9, 10, 30522] 
            scores0 = torch.ones(scores_shape, device=lidar_values['pixel_values'].device) 

            seq_scores_shape = [10] 
            seq_scores0 = torch.zeros(seq_scores_shape, device=lidar_values['pixel_values'].device) 
             

            
            for e, expert in enumerate(self.experts):

                if batch_size == 1: # evaluation case
                    if float(CUSTOM_GATE[0][e]) == 0: # if not active expert
                        all_outputs.append(scores0)
                        all_sequences_scores.append(seq_scores0)
                        continue 

    
                out = expert.generate(
                        pixel_values=lidar_values['pixel_values'],
                        lidar_values=lidar_values,
                        points=None,
                        max_length=ID_MAX_LENGTH,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        eos_token_id=None,
                        pad_token_id=0,
                        bos_token_id=2,
                        renormalize_logits=False,
                        early_stopping=False, #True,#
                        prefix_allowed_tokens_fn=restrict_decode_vocab_v3,
                        return_dict_in_generate=True,                
                        output_scores = True,
                )
                
                scores = torch.stack(out.scores, dim=0)  # [seq_len, batch*beams, vocab]
                scores = torch.exp(scores)               # convert from log-probs to probs
                all_outputs.append(scores)
                
                #all_sequences_scores.append(out.sequences_scores.reshape([-1, num_beams]))
                seq_score = out.sequences_scores
                all_sequences_scores.append(seq_score)
                

            all_outputs2 = torch.stack(all_outputs, dim=0)

            gw = gate_weights.transpose(0, 1)  # → [4, 8]
            gw = gw.unsqueeze(1).unsqueeze(-1)  # → [4, 1, 8, 1]
            gw = gw.repeat(1, all_outputs2.shape[1], 1, 1)
            gw = gw.repeat(1, 1, num_beams, 1)

            mixed_probs = torch.sum(all_outputs2 * gw, dim=0)
            mixed_output = torch.log(mixed_probs + 1e-9)


            # pick tokens
            token_ids = mixed_output.argmax(dim=-1)

            #mixed_id = [ self.tokenizer.decode(x, skip_special_tokens=True) for x in token_ids2 ] # 640
            #print( "mixed_id = ",mixed_id)
            if batch_size == 1:
                #################################################################
                # sequences_scores ponderatio
                seq_scores = torch.stack(all_sequences_scores, dim=0)  # [num_experts, num_beams]
                gw_seq = gate_weights[0]  # [num_experts], since batch=1
                
                combined_probs = torch.sum(torch.exp(seq_scores) * gw_seq.unsqueeze(-1), dim=0)  # [num_beams]
                combined_log_scores = torch.log(combined_probs + 1e-9)  # [num_beams]
                #################################################################
                return {"sequences": token_ids.T, "scores": mixed_output, "gate_weights": gate_weights, "sequences_scores": combined_log_scores }
            return {"sequences": token_ids.T, "scores": mixed_output, "gate_weights": gate_weights}

        def generate_top1(self, x=None, pixel_values=None, lidar_values=None, **kwargs):
            
            if x is None:
                x = self._make_gate_input(lidar_values)

            dispatch_tensor, combine_tensor, gate_loss, CUSTOM_GATE = self.gate(x)

            # 1. pick active expert
            active = torch.argmax(CUSTOM_GATE[0]).item()
            expert = self.experts[active]
        
            # 2. run only that expert
            out = expert.generate(
                pixel_values=lidar_values['pixel_values'],
                lidar_values=lidar_values,
                points=None,
                max_length=ID_MAX_LENGTH,
                num_beams=10,
                num_return_sequences=10,
                eos_token_id=None,
                pad_token_id=0,
                bos_token_id=2,
                renormalize_logits=False,
                early_stopping=False,
                prefix_allowed_tokens_fn=restrict_decode_vocab_v3,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
            # 3. No mixing necessary
            scores = torch.stack(out.scores, dim=0)
            mixed_output = scores   # Only 1 expert
        
            # 4. decode
            token_ids = mixed_output.argmax(dim=-1)
        
            return {
                "sequences": token_ids.T,
                "scores": mixed_output,
                "gate_weights": CUSTOM_GATE,
                "sequences_scores": out.sequences_scores,
            }
        
        def generate_top1_V2(self, x=None, pixel_values=None, lidar_values=None, **kwargs):
            if x is None:
                x = self._make_gate_input(lidar_values)
        
            _, _, _, CUSTOM_GATE = self.gate(x)
            active = torch.argmax(CUSTOM_GATE[0]).item()
            expert = self.experts[active]
        
            out = expert.generate(
                pixel_values=lidar_values['pixel_values'],
                lidar_values=lidar_values,
                points=None,
                max_length=ID_MAX_LENGTH,
                num_beams=10,
                num_return_sequences=10,
                pad_token_id=0,
                bos_token_id=2,
                eos_token_id=3,
                renormalize_logits=False,
                early_stopping=False,
                prefix_allowed_tokens_fn=restrict_decode_vocab_v3,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
            return {
                "sequences": out.sequences,              # ✅ correct beams
                "scores": out.scores,
                "gate_weights": CUSTOM_GATE,
                "sequences_scores": out.sequences_scores # ✅ correct ranking
            }


    #############################################################################
    # moe_model
    #############################################################################
    from transformers import get_scheduler
    
    moe_model = MoE_DSI(input_dim=256, experts=transformer_list, tokenizer=tokenizer, expert_labels=expert_labels).to(device)

    moe_model.eval()
    optimizer = torch.optim.Adam(moe_model.gate.parameters(), lr=1e-1)

    num_training_steps = 1000
    
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer,
        num_warmup_steps=0,  
        num_training_steps=num_training_steps,
    )


    for name, p in moe_model.named_parameters():
        if p.requires_grad:
            print("moe_model.named_parameters requiring grad ", name)

    logger.info("============== FULL NETWORK STATE =================")
    for name, param in moe_model.named_parameters() : logger.info(name + "\t =>" + str(param.requires_grad)) 
    logger.info("============== FREE NETWORK STATE =================")

    ### ====== Print active layers git =========
    Print_active_layers_git(args, logger, moe_model, model_name)  
    logger.info("============== NETWORK STATE =================")
    #############################################################################

    #print("len train_subset", len(train_subset))

    #############################################################################
    ### === Training / Evaluation ===
    ##############################################################################
    

    if do_train:

        if cfg['DATA_CONFIG']['DATASET'] == "LHD_dataset":
            is_training = False
            
            train_indices = all_train_indices
            val_indices = all_train_indices

        #############################################################################
        # Preload required data for compute metrics only one time 
        #############################################################################
        def load_json(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        
        label_mapping = {}
        label_mapping_val = {}
        if args.local_rank == 0 :
            print("train_set.labeltype ",train_set.labeltype)
            
        if train_set.labeltype in {"hilbert"}:
            #label_mapping = load_json(sequence_path + "/hilbert_12_pad.json")
            #label_mapping_val = load_json(sequence_path + "/hilbert_12_pad_val.json")
            #print("label_mapping_path ", sequence_path + "/hilbert_12_pad_val.json")

            label_mapping = load_json(sequence_path + "/hilbert_13_pad.json")
            label_mapping_val = load_json(sequence_path + "/hilbert_13_pad_val.json")
            print("label_mapping_path ", sequence_path + "/hilbert_13_pad_val.json")
            
            print("len(label_mapping)", len(label_mapping))
        #############################################################################

            
        positions_database = train_set.positions_database # no need to shift --> use to compute metrics
        
        print("len(positions_database)", len(positions_database))
        print("len(set(lid)) " ,len(set(lid))) # 20900
        print("len(label_mapping.keys()) ", len(label_mapping.keys()))  # 20892
        print("diff ", set(lid).difference(label_mapping.keys())  )  #{'03056', '54175', '04577', '04451', '33475', '07829', '03130', '54142'}

        
        train_subset = torch.utils.data.Subset(train_set, train_indices)
        if args.local_rank == 0 :
            print("train_indices ", len(train_indices))
        del train_set, train_indices
        
        val_subset = torch.utils.data.Subset(eval_set, val_indices)
        if args.local_rank == 0 :
            print("val_indices ", len(val_indices))
        del eval_set, val_indices
        gc.collect()  # Force garbage collection

        #############################################################################
        # shuffling
        #############################################################################
        indices = torch.randperm(len(train_subset))
        train_subset_shuffled = torch.utils.data.Subset(train_subset, indices)

        indices = torch.randperm(len(val_subset))
        val_subset_shuffled = torch.utils.data.Subset(val_subset, indices)
        #############################################################################
     
        if args.local_rank == 0 :
            print("train_subset_shuffled ", len(train_subset))
            print("train_subset_shuffled ", train_subset_shuffled[0]['frame_id'], train_subset_shuffled[1]['frame_id'], train_subset_shuffled[2]['frame_id'])
            print("val_subset ", len(val_subset))
            print("val_subset ", val_subset[0]['frame_id'], val_subset[1]['frame_id'], val_subset[2]['frame_id'])
 
        
        # resume
        previous_model_path = args.resume_from_checkpoint

        new_train_batch_size = ori_train_batch_size
        new_eval_batch_size = ori_eval_batch_size
        
        if args.local_rank == 0 :
            print("new_train_batch_size ", new_train_batch_size)
            print("new_eval_batch_size ", new_eval_batch_size)
        new_training_args = replace(training_args, 
                                    per_device_train_batch_size=new_train_batch_size,
                                    per_device_eval_batch_size=new_eval_batch_size)

        save_file_name = args.save_hit_file # new for naming file hit score

        if args.local_rank == 0 :
            logger.info("  ")
            logger.info(" ======= START TRAINING ========= ")
            logger.info("train_set_len:" + str(len(train_subset)))

        trainer = DSITrainer(
                model=moe_model,
                tokenizer=tokenizer,
                args=new_training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                data_collator=data_collator,
                compute_metrics=make_compute_metrics(tokenizer, logger, args.local_rank, positions_database, label_mapping, label_mapping_val, save_file_name),
                restrict_decode_vocab=restrict_decode_vocab_v3,
                LIK=LIK,
                optimizers=(optimizer, lr_scheduler), 
                #callbacks=[EarlyStoppingCallback(metric_name="eval_Hits@1", threshold=0.99)],  # Custom callback
                id_max_length=ID_MAX_LENGTH
            ) 

        is_trained = False
        if not os.path.isdir(previous_model_path) :
            if args.local_rank == 0 :
                print("train from scratch : ",previous_model_path)
            trainer.train()
            is_trained = True
        else :
            if args.local_rank == 0 :
                print("resume_from_checkpoint : " + previous_model_path)
            trainer.train(resume_from_checkpoint=previous_model_path)
            is_trained = True
   
    
    if do_eval  : 

        """
        eval_indicesa = load_set_ids("id_zone_A_dsi_eval_list.json", sequence_path)
        eval_indicesb = load_set_ids("id_zone_B_dsi_eval_list.json", sequence_path)
        eval_indicesc = load_set_ids("id_zone_C_dsi_eval_list.json", sequence_path)
        eval_indicesd = load_set_ids("id_zone_D_dsi_eval_list.json", sequence_path)
        
        eval_indicesa = load_set_ids("id_small_list_A.json", sequence_path)
        eval_indicesb = load_set_ids("id_small_list_B.json", sequence_path)
        eval_indicesc = load_set_ids("id_small_list_C.json", sequence_path)
        eval_indicesd = load_set_ids("id_small_list_D.json", sequence_path)
        
        all_eval_indices = eval_indicesa + eval_indicesb + eval_indicesc + eval_indicesd
        """

        all_eval_indices1 = load_set_ids("id_dsi_eval_list.json", root_path1)
        all_eval_indices2 = load_set_ids("id_dsi_eval_list.json", root_path2)
        print(len(all_eval_indices1))
        print(len(all_eval_indices2))     

        
        eval_indices1 = all_eval_indices1
        eval_subset1 = torch.utils.data.Subset(eval_set, eval_indices1)
        print("len(eval_subset1) ", len(eval_subset1))

           
        eval_indices2 = all_eval_indices2
        eval_subset2 = torch.utils.data.Subset(eval_set2, eval_indices2)
        print("len(eval_subset2) ", len(eval_subset2))


       
        
        if args.local_rank == 0 :
            print("start eval")
        eval_log3dnet(moe_model, eval_subset1, eval_set, eval_indices1, eval_subset2, eval_set2, eval_indices2, data_collator, tokenizer, cfg, checkpoint_dir, checkp_to_eval, prefix_dict, LIK, ID_MAX_LENGTH)



if __name__ == '__main__':
    main()






