#!/bin/bash

#SBATCH --mail-user=lm19073
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gpuTest
#SBATCH --output=gpuTest_%j.out
#SBATCH --error=gpuTest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --mem=15G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load CUDA
module load languages/anaconda3/3.7
module load tools/git/2.18.0
export LD_LIBRARY_PATH=/mnt/storage/software/languages/anaconda/Anaconda3-2019-3.7/lib:$LD_LIBRARY_PATH 

echo 'Running on: '$HOSTNAME

srun python train.py --cfg ../scratch/ego-hand/yolov3-hand-anchors.cfg --data ../scratch/ego-hand/ego-hand.data --epochs 100 --batch-size 4 --weights '' --nosave && mv results.txt egohand_egoanchor.txt
