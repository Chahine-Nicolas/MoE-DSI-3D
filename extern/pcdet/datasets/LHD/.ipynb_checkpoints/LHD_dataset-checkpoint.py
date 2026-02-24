#####################################################################################
#####################################################################################
#####################################################################################

import numpy as np
from ..dataset import DatasetTemplate
from ..dsi_datasets import DSIDatasets
import glob
import os
from PIL import Image
import pathlib
import torch
import fnmatch
import random
import re
import json

#####################################################################################
# opti
#####################################################################################
class LHD_dataset(DatasetTemplate,DSIDatasets):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        super(DatasetTemplate, self).__init__(dataset_cfg=dataset_cfg)

        self.training = training
        self.kitti_infos = []
        self.eval_seq = dataset_cfg['SEQ']
        self.eval_seq_str = dataset_cfg['SEQ']
        self.root_path = self.root_path / self.eval_seq_str
        self.include_lhd_data()
        self.dataset = dataset_cfg['DATASET']

        # Model components
        self.tokenizer = None
        self.image_processor = None
        self.ID_MAX_LENGTH = None
        self.do_self_eval = dataset_cfg['DO_SELF_EVAL']
        self.train_only_revisited = dataset_cfg['TRAIN_ONLY_REVISITED']

        # Distance Criteria
        self.revisit_criteria = 3
        self.skip_time = 30

        # Labels
        self.hierarchical_label = {}
        self.gps_label = {}

        # Contrastive Learning Parameters
        self._load_contrastive_data(revisit_criteria = 3, revisit_criteria_extended = 3) # change here the negative distance threshold

        # Load dataset files
        self.files = self._load_files()

        # Load poses
        with open(self.root_path / 'poses_grid2.json') as f:
            poses = json.load(f)
        self.positions_database = poses

        # Load hierarchical labels
        self.hierarchical_label, self.inv_hierarchical_label = self._load_json_labels('hierarchical_indexing.json')

        # Load GPS labels
        self.gps_label, self.inv_gps_label = self._load_json_labels('dict_gps_2_label_v2.json')

        # Load ground truth info if not self-evaluating matching
        #if not self.do_self_eval:
            #self.load_gt_infos(self.root_path)

    def _load_contrastive_data(self, revisit_criteria = 3, revisit_criteria_extended = 20,  skip_time = 0):
        """
        Loads positive and negative sequences for contrastive learning.
        """
        tuple_dir = self.root_path
        #lhd_pos_json = f'positive_sequence_D-{revisit_criteria}_T-{skip_time}.json'
        #lhd_neg_json = f'positive_sequence_D-{revisit_criteria_extended}_T-{skip_time}.json'
        
        #lhd_pos_json = 'dict_hardest_pos.json'
        #lhd_neg_json = 'revisits_lidar_full_train_list.json'

        #lhd_pos_json = 'dsi_revisits_lhd_1m.json'
        lhd_pos_json = 'dsi_revisits_lhd_itself.json' 
        lhd_neg_json = 'dsi_revisits_lhd_59mt.json'
        lhd_neg_json = 'dsi_revisits_lhd_itself.json'
        

        self.dict_3m = json.load(open(tuple_dir / lhd_pos_json, "r"))
        self.dict_20m = json.load(open(tuple_dir / lhd_neg_json, "r"))  #define negative distance threshold
        
    
    def _load_files(self):
        """
        Loads file information for queries, positives, and negatives.
        """
        
        f = open(str(self.root_path) + "/full_list.json") 
        fnames = json.load(f)
        f.close()
        
        # print("before load ", str(self.root_path) + '/bin/*.bin')
        #fnames = glob.glob(str(self.root_path) + '/bin/*.bin')
        #inames = sorted(os.path.split(fname)[-1][:-4] for fname in fnames)
        inames = sorted(fnames)


        #f = open(str(self.root_path) + "/zone_A_dsi_train_list.json") 
        f = open(str(self.root_path) + "/dsi_train_list.json") 
        print("negative listing", f)
        fnames = json.load(f)
        f.close()
        inames2 = sorted(fnames)
        
        
        self.lhd_seq_lens = {str(self.eval_seq): len(fnames)}
        files = []
        for query_id in inames:
            positives = self.get_positives(self.eval_seq, query_id)
            negatives = self.get_negatives(self.eval_seq, query_id, inames2)
            
            if len(negatives) > 10000: # keep randommly only 1000 negatives to save memory
                negatives = random.sample(negatives, 10000)
            files.append((self.eval_seq, query_id, positives, negatives))
        
        return files



    def _load_json_labels(self, filename):
        """
        Loads JSON-based labels such as hierarchical or GPS labels.
        """
        path = self.root_path / filename
        if path.is_file():
            with open(path, "r") as json_file:
                labels = json.load(json_file)
            return labels, {v: k for k, v in labels.items()}
        return {}, {}

    def include_lhd_data(self):
        """
        Loads LiDAR HD dataset files.
        """
        if self.logger:
            self.logger.info('Loading LiDAR HD dataset')

        """
        self.kitti_infos = sorted(
            self.client.list_dir_or_file(self.root_path / "bin", list_dir=False, recursive=True, suffix='.bin'),
            key=lambda s: int(re.search(r'\d+', s).group())
        )
        """
        f = open(str(self.root_path) + "/full_list.json") 
        self.kitti_infos  = json.load(f)
        f.close()

        if self.logger:
            self.logger.info(f'Total samples for LiDAR HD dataset: {len(self.kitti_infos)}')

        
    def get_lidar(self, idx):
        """
        Loads LiDAR data.
        """
        lidar_file = self.root_path / 'bin' / str(idx)
        return self.client.load_to_numpy(str(lidar_file), dtype=np.float32).reshape(-1, 4)


    ###################################################################################################
    ## ajout du chargement des nuages similaire ou non
    ###################################################################################################
    def get_positives(self, sq, index):
        """
        Retrieves positive samples for a given sequence and index.
        Filters out indices that are divisible by 5 to avoid val/eval set contamination.
        """
        sq = self.dict_3m
        if index in sq:
            positives = sq[index]
            if index in positives and False:
                positives.remove(index)
        else:
            positives = [index]
        
        return positives



    def get_negatives(self, sq, index, inames):
        """
        Retrieves negative samples for a given sequence and index.
        Ensures negatives are disjoint from positives.
        """
        
        # classic negatives
        all_ids = set(inames)
        #sq = self.dict_20m
        if index in self.dict_20m:
            neg_set_inv = set(self.dict_20m[index])
        else:
            neg_set_inv = set([])
        
        neg_set = all_ids.difference(neg_set_inv)
        
        negatives = list(neg_set)
        if index in negatives and False:
            negatives.remove(index)
        """
        sq = self.dict_20m
        if index in sq:
            negatives = sq[index]
            if index in negatives and False:
                negatives.remove(index)
        else:
            negatives = [index]
         """
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids, inames):
        # Dissimillar to all pointclouds in triplet tuple.
        
        all_ids = set(inames)
        neighbour_ids = sel_positive_ids
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(drive_id, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        assert len(
            possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]
    
    
    def load_tensor_just_one(self,nn,lab) :
        xx = []
        fname = self.root_path /  lab /  (nn+'.pt')
        if os.path.isfile(fname) : 
            xx1 = torch.load(fname, map_location='cuda')
            xx.append(xx1)
        else :
            print("TENSOR NOT FOUND")
            print(fname)
            return None
        return torch.stack(xx)

    
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

    
    def __getitem__(self, index):
        # Handle merging of iterations to one epoch
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
    
        # Load LiDAR path and extract metadata
        if isinstance(index, int):
            lidar_path = os.path.basename(self.kitti_infos[index])
        elif isinstance(index, str):
            lidar_path = os.path.basename(index)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        path_split = str(lidar_path).split('/')
    
        input_dict = {'frame_id': path_split[0][:-4]}
        input_dict['frame_id_desc'] = self.load_tensor_just_one(input_dict['frame_id'],"256_desc_2025-06-23_11-22-13_run_0_4").unsqueeze(1).to(dtype=torch.float32)

        ### Contrastive
        #main_id = self.get_id(index)
        
        drive_id, query_id, positive_ids, negative_ids = self.files[index]

        # drive_id, query_id, positive_ids, negative_ids[0]

        # Select contrastive samples
        sel_positive_ids = random.sample(positive_ids, min(len(positive_ids), 1))
        del positive_ids
        
        sel_negative_ids = random.sample(negative_ids, min(len(negative_ids), 9))
        del negative_ids
        
        # Get another distinct negative sample

        # if all dataset
        f = open(str(self.root_path) + "/full_list.json") 
        fnames = json.load(f)
        f.close()
        inames = sorted(fnames)

        # if subzone
        #f = open(str(self.root_path) + "/zone_A_dsi_train_list.json")
        f = open(str(self.root_path) + "/dsi_train_list.json") 
        fnames = json.load(f)
        f.close()
        inames = sorted(fnames)

        
        other_neg_id = self.get_other_negative(drive_id, query_id, sel_positive_ids, sel_negative_ids, inames)
    
        ### Image Processing
        image_path = './extern/proxy.jpg'
        image = Image.open(image_path).convert('RGB')
    
        if self.image_processor:
            input_dict['pixel_values'] = self.image_processor(images=image, return_tensors="pt")['pixel_values'].contiguous()

        ### Load LiDAR Data (if required)
        if "points" in get_item_list:
            input_dict['points'] = self.get_lidar(lidar_path)

            input_dict.update(self.get_dict_dsi(index))  # Merge dictionaries

            # Extract LiDAR values for `lidar_values`
              # Initialize dictionary
            
            lidar_values = {}
            #lidar_values['batch_size'] = 1
            #lidar_values['points'] = torch.tensor(input_dict['points'], dtype=torch.float32) 
            
            
            # Tokenization (if applicable)
            if self.tokenizer:
                res = self.tokenizer(
                    input_dict['labels'],
                    padding="max_length",
                    return_tensors="pt",
                    truncation='only_first',
                    max_length=self.ID_MAX_LENGTH
                )
                
                input_dict.update({
                    'input_ids': res.input_ids[0],
                    'attention_mask': res.attention_mask[0],
                    #'id_pcd_positif':  ['velodyne', f'{sel_positive_ids[0]:06d}.bin'],
                    #'id_pcd_negatif':  ['velodyne', f'{sel_negative_ids[0]:06d}.bin'],
                    #'other_id_pcd_negatif':  ['velodyne', f'{other_neg_id:06d}.bin']
                    
                    'id_pcd_positif':  str(sel_positive_ids).split('/')[-1][:-6],
                    'id_pcd_negatif':  str(sel_negative_ids).split('/')[-1][:-6],
                    'other_id_pcd_negatif': str(other_neg_id).split('/')[-1][:-4]
                })
                
                input_dict['id_pcd_positif_desc'] = self.load_tensor_just_one(input_dict['id_pcd_positif'],"256_desc_2025-06-23_11-22-13_run_0_4").unsqueeze(1).to(dtype=torch.float32)
                input_dict['id_pcd_negatif_desc'] = self.load_tensor_just_one(input_dict['id_pcd_negatif'],"256_desc_2025-06-23_11-22-13_run_0_4").unsqueeze(1).to(dtype=torch.float32)
                input_dict['other_id_pcd_negatif_desc'] = self.load_tensor_just_one(input_dict['other_id_pcd_negatif'],"256_desc_2025-06-23_11-22-13_run_0_4").unsqueeze(1).to(dtype=torch.float32)

 
                
        # Prepare final data dictionary
        data_dict = self.prepare_data(data_dict=input_dict) # add 'transformation_3d_list': ['random_world_flip', 'random_world_rotation', 'random_world_scaling'], 'transformation_3d_params': {'random_world_flip': [], 'random_world_rotation': 0.0, 'random_world_scaling': 1.0}, 'use_lead_xyz': True, 'transformation_2d_list': [], 'transformation_2d_params': {}}
        
        # Add LIDAR data
        #data_dict['points'] = input_dict['points']
        data_dict['lidar_values'] = lidar_values  

        return data_dict
