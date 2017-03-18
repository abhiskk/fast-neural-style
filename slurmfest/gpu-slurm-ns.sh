#!/bin/bash
#
#SBATCH --job-name=ak_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1
#SBATCH --time=23:00:00
#SBATCH --mem=50000
#SBATCH --output=ak_experiment_%A.out
#SBATCH --error=ak_experiment_%A.err
#SBATCH --mail-user=ak6179@nyu.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python3/intel/3.5.3
module load cuda/8.0.44

python3 -m pip install -U pip setuptools --user
python3 -m pip install cmake --user
python3 -m pip install numpy --upgrade --user
python3 -m pip install scipy --upgrade --user
python3 -m pip install git+https://github.com/pytorch/pytorch.git@master --upgrade --user
python3 -m pip install torchvision --upgrade --user

cd /scratch/ak6179/fast-neural-style

python3 -u neural_style/neural_style.py --batch-size 4 --epochs 2 --dataset /scratch/ak6179/mscoco-train-2014 --cuda 1 --vgg-model vgg-model --style-image images/style-images/mosaic.jpg --checkpoint-dir checkpoints --train 1 > experiments/fast-ns-COCO.gpu.21.log
