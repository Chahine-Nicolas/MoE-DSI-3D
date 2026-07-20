## Overview

**MoE-DSI-3D** is an adaptation of Mixture-of-Experts (MoE) architecture to our [DSI-3D model](https://github.com/Chahine-Nicolas/DSI-3D/tree/main)

![plot](https://github.com/Chahine-Nicolas/MoE-DSI-3D/blob/main/__assets__/img/moe2.PNG?raw=true)


# NEWS

- [2026-02] Code release
- [2026-07] Model weights

# MoE-DSI-3D Checkpoint

[Download here](https://huggingface.co/Chahine-Nicolas/MoE-DSI-3D/tree/main)

### Paris_East Experts

| Expert | File |
|--------|------|
| A | `git_hilbert_A0.zip` |
| B | `git_hilbert_B0.zip` |
| C | `git_hilbert_C0.zip` |
| D | `git_hilbert_D0.zip` |

### Paris_West Experts

| Expert | File |
|--------|------|
| A | `git_hilbert_A1.zip` |
| B | `git_hilbert_B1.zip` |
| C | `git_hilbert_C1.zip` |
| D | `git_hilbert_D1.zip` |
| E | `git_hilbert_E1.zip` |

**`Gate_Model.zip`:** 
Contents:

| Model | Description |
|-------|-------------|
| `gate_east` | Routes inputs to the Paris_East experts. |
| `gate_west` | Routes inputs to the Paris_West experts. |
| `gate_east_west` | Selects between the Paris_East and Paris_West expert groups. |

Fine-tuned LoGG3D-Net on **LiDAR HD**.
- `LoGG3D-Net (Re-trained on LHD).zip`

## Tokenizer

Tokenizer vocabulary optimized for Hilbert-curve indexing of LiDAR HD.
- `transformers_vocab.zip`


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
