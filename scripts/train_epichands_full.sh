#!/bin/bash

#SBATCH --mail-user=lm19073
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=train-epichands-full
#SBATCH --output=train-epichands-full_%j.out
#SBATCH --error=train-epichands-full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

module load CUDA
module load languages/anaconda3/3.7
module load tools/git/2.18.0
export LD_LIBRARY_PATH=/mnt/storage/software/languages/anaconda/Anaconda3-2019-3.7/lib:$LD_LIBRARY_PATH 

cd yolov3
git pull

srun python train.py --cfg ../scratch/epichands/yolov3-spp-epichands.cfg --data ../scratch/epichands/epichands.data --epochs 300 --batch-size 16 --weights '' --nosave && mv results.txt epichands_varied_size.txt
