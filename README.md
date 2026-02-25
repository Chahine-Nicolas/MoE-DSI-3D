


# Experts training
Set the path to your dataset in kitti_lhd_hilbert.yaml

```highlight
DATA_PATH = ../lidarhd_east
```

Experts training is defined with :

```highlight
job_train_lhd.slurm
```

inside main_80_20.py

you will need to affect sub_part = True for a MoE-DSI-3D expert training or sub_part = False for a DSI-3D training 

For expert training, you will need to set train_indices_path and val_indices_path to the corresponding area (A, B, C, D, and E).
example :

```highlight
train_indices_path = "id_zone_A_dsi_train_list.json"
val_indices_path = "id_zone_A_dsi_val_list.json"
eval_indices_path = "id_zone_A_dsi_eval_list.json"
```

# Gate training


Train and evaluate Gate East:
```highlight
python train_relu_lhd_multilabel.py
```

Train and evaluate Gate West:
```highlight
python train_relu_lhd_multilabel_OUEST.py
```

Train and evaluate Gate East + West:
```highlight
python train_relu_lhd_multilabel_EST_OUEST.py
```

# MoE-DSI-3D evaluation


in kitti_lhd_hilbert.yaml set DATA_PATH to lidarhd_v2 then :
```highlight
sbatch job_eval_lhd_moe_east.slurm
```

in kitti_lhd_hilbert.yaml set DATA_PATH to lidarhd_v3 then :
```highlight
sbatch job_eval_lhd_moe_west.slurm
```

in kitti_lhd_hilbert.yaml set DATA_PATH to lidarhd_v4 then :
```highlight
sbatch job_eval_lhd_moe_east_west.slurm
```
