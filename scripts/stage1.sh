#!/bin/bash
#SBATCH --job-name=stage1
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/stage1%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/stage1%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate

cd /user/gaad2403/Applied
/fs/dss/home/gaad2403/negation_env/bin/python -u src/stage1/run.py --config configs/stage1.yaml