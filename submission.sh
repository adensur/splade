#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --job-name=splade-dist
#SBATCH --mem=0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --partition=dev
#SBATCH --time=800:00:00
#SBATCH --error=logs/splade_citr7.err
#SBATCH --output=logs/splade_citr7.out

set -e # set fail on fail
echo "Slurm job id "$SLURM_JOB_ID
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/traindata/maksim/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/traindata/maksim/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/traindata/maksim/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/traindata/maksim/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate splade

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="config_citr3"
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Set local rank environment variable for proper GPU assignment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400

srun bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
    -m splade.hf_train \
    config.checkpoint_dir=experiments/splade_citr7/checkpoint \
    config.index_dir=experiments/splade_citr7/index \
    config.out_dir=experiments/splade_citr7/out"