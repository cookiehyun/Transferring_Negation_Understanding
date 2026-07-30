#!/bin/bash
#SBATCH --job-name=stage2
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/stage2_llm_mcq%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/stage2_llm_mcq%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate

cd /user/gaad2403/Applied
/fs/dss/home/gaad2403/negation_env/bin/python -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config configs/stage2.yaml \
  --retrieval_csv outputs/negbench/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv \
  --coco_root outputs/coco/images/val2017 \
  --extractor llm \
  --n_images -1 \
  --lambdas 0.1,0.2,0.25,0.3,0.35,0.5,0.8,1.2,1.9