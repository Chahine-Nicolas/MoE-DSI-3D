#!/bin/bash                                                                   
#SBATCH --partition=gpu_p13                                                  
#SBATCH --nodes=1                                                               
#SBATCH --ntasks-per-node=1    
#SBATCH --qos=qos_gpu-t4             
#SBATCH --time=100:00:00                                 
#SBATCH --gres=gpu:1                                                           
#SBATCH --cpus-per-task=10                                                    
#SBATCH --hint nomultithread                                                    
#SBATCH --mail-user user@mail.com                          
#SBATCH --job-name=train_ouest                      
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err                     

# ============================================================
# Bash / Slurm setup
# ============================================================

echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Job name     : ${SLURM_JOB_NAME}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "============================================================"

# ============================================================
# Environment
# ============================================================
                          
module purge
module load anaconda-py3/2023.03
conda activate MOE_DSI_3D

export PYTHONUSERBASE=env_path
export LOG3DNET_DIR=LoGG3D_Net_code_path

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# Optional
export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)

# ============================================================ 
# Experiment 
# ============================================================
MODEL_NAME="git" 
LABEL_MODE="hilbert" # "gps", "label", "hierarchical"

DO_USE_SOP=true 

DO_TRAIN=true
DO_EVAL=false
DO_DUMP_DICT_GT=false
DATASET_LEN=-1


ID_MAX_LENGTH=10
NUM_TRAIN_EPOCH=100
BATCH_SIZE_TRAIN=256
BATCH_SIZE_EVAL=64
EVAL_STEP=500
SAVE_STEPS=100

LEARNING_RATE=5e-4 
ADAM_EPSILON=1e-5 
ADAM_BETA1=0.90 
ADAM_BETA2=0.999

WARMUP_STEPS=100 
LR_SCHEDULER_TYPE="linear" 
LOGGING_STEPS=1 
RESUME_CHECKPOINT=false

# ============================================================ 
# Paths 
# ============================================================

CONFIG_NAME="config_lhd_${LABEL_MODE}.yaml" 
CHECKPOINT="../ckpts/gd_mae_finetune_kitti.pth" 
EXTRA_TAG="${MODEL_NAME}_${LABEL_MODE}_train_ouest" 
OUTPUT_DIR="${WORK}/checkpoints/${EXTRA_TAG}" 
EVAL_CHECKPOINT="../checkpoints/${EXTRA_TAG}/"

echo "============================================================" 
echo "Experiment configuration" 
echo "============================================================" 
echo "Model : ${MODEL_NAME}" 
echo "Label mode : ${LABEL_MODE}" 
echo "Config : ${CONFIG_NAME}" 
echo "Checkpoint : ${CHECKPOINT}" 
echo "Output directory : ${OUTPUT_DIR}" 
echo "Train : ${DO_TRAIN}" 
echo "Eval : ${DO_EVAL}" 
echo "Train batch size : ${BATCH_SIZE_TRAIN}" 
echo "Eval batch size : ${BATCH_SIZE_EVAL}"
echo "Learning rate : ${LEARNING_RATE}" 
echo "Epochs : ${NUM_TRAIN_EPOCH}" 
echo "============================================================"


# ============================================================ 
# srun 
# ============================================================

python main_80_20.py \
       --launcher none \
       --cfg_file ${CONFIG_NAME} \
       --save_hit_file hit_scores.txt \
       --workers 1 \
       --model_name ${MODEL_NAME} \
       --dataset_train_len ${DATASET_LEN} \
       --dataset_eval_len ${DATASET_LEN} \
       --pretrained_model ${CHECKPOINT} \
       --per_device_train_batch_size ${BATCH_SIZE_TRAIN} \
       --per_device_eval_batch_size ${BATCH_SIZE_EVAL} \
       --save_to_file \
       --output_dir=${OUTPUT_DIR} \
       --adam_epsilon=${ADAM_EPSILON} \
       --adam_beta1=${ADAM_BETA1} \
       --adam_beta2=${ADAM_BETA2} \
       --num_train_epochs ${NUM_TRAIN_EPOCH} \
       --learning_rate=${LEARNING_RATE} \
       --eval_steps ${EVAL_STEP} \
       --evaluation_strategy steps \
       --use_sop ${DO_USE_SOP} \
       --save_steps ${SAVE_STEPS} \
       --do_train ${DO_TRAIN} \
       --do_eval ${DO_EVAL} \
       --id_max_length ${ID_MAX_LENGTH} \
       --do_dump_dict_gt ${DO_DUMP_DICT_GT} \
       --resume_from_checkpoint ${RESUME_CHECKPOINT} \
       --eval_checkpoint ${EVAL_CHECKPOINT}  \
       --extra_tag ${EXTRA_TAG} \
       --warmup_steps ${WARMUP_STEPS} \
       --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
       --logging_steps ${LOGGING_STEPS} 

echo "============================================================" 
echo "Job finished successfully: $(date)" 
echo "============================================================"