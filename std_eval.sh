#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Northstar
#SBATCH --nodes=1
#SBATCH --time=300:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --output=RDDM_AAAI24/PUBLIC/eval-logs/standard_eval_RDDM.out
#SBATCH --error=RDDM_AAAI24/PUBLIC/eval-logs/standard_eval_RDDM.err

export TMPDIR=/home/21ds94/tmp

# load module
source /home/21ds94/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu

python RDDM_AAAI24/PUBLIC/std_eval.py