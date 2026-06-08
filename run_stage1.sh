#!/bin/bash
#SBATCH --job-name=stage1
#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=mpcg[003-006]
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/%j.err

module load hpc-env/13.1 Python/3.11.3-GCCcore-13.1.0
source /fs/dss/home/gaad2403/negation_env/bin/activate

cd /user/gaad2403/Applied
python stage1.py
