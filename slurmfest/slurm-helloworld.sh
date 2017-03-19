#!/bin/bash
#
#SBATCH --job-name=ak_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p1080:1
#SBATCH --time=23:00:00
#SBATCH --mem=200000
#SBATCH --output=ak_experiment_%A.out
#SBATCH --error=ak_experiment_%A.err
#SBATCH --mail-user=ak6179@nyu.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python3/intel/3.5.3

python3 -m pip install -U pip setuptools --user
python3 -m pip install cmake --user
python3 -m pip install numpy --upgrade --user
python3 -m pip install scipy --upgrade --user
python3 -m pip install git+https://github.com/pytorch/pytorch.git@master --upgrade --user
python3 -m pip install torchvision --user

cd /scratch/ak6179/testing

python3 -u helloworld.py > helloworld.log
