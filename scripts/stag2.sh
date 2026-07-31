#!/bin/bash
#SBATCH --job-name=llama_conclip_ccneg
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/llama_conclip_ccneg_%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/llama_conclip_ccneg_%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate
cd /user/gaad2403/Applied
PYTHON=/fs/dss/home/gaad2403/negation_env/bin/python

CFG=configs/stage2.yaml

echo "########## Llama + ConCLIP + hybrid, CC-Neg ##########"
$PYTHON -u src/stage2/run_ccneg_content_aware_eval.py \
  --config $CFG \
  --extractor hybrid --clip_backend conclip \
  --lambdas 0.1,0.3,0.5,0.8,1.0,1.5,2.0,2.5,3.0

echo "########## Done ##########"
