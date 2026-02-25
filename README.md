
# Experts training



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
