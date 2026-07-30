#!/bin/bash
#SBATCH --job-name=fit_W
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/fit_W_%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/fit_W_%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate
cd /user/gaad2403/Applied
/fs/dss/home/gaad2403/negation_env/bin/python -u src/stage2/fit_concept_correction.py
