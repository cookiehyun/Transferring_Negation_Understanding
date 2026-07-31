#!/bin/bash
#SBATCH --job-name=qwen14b_test
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/qwen14b_test_%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/qwen14b_test_%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate
cd /user/gaad2403/Applied
PYTHON=/fs/dss/home/gaad2403/negation_env/bin/python

MCQ_TEST=outputs/negbench/COCO_val_mcq_test.csv
RETRIEVAL_TEST=outputs/negbench/COCO_val_negated_retrieval_test.csv
COCO_ROOT=outputs/coco/images/val2017
CFG=configs/stage2_qwen14b.yaml

echo "########## Qwen14B-llm x MCQ, lambda=0.86 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor llm --n_rows -1 --lambdas 0.86

echo "########## Qwen14B-hybrid x MCQ, lambda=0.93 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor hybrid --n_rows -1 --lambdas 0.93

echo "########## Qwen14B-llm x Retrieval, lambda=0.49 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor llm --n_images -1 --lambdas 0.49

echo "########## Qwen14B-hybrid x Retrieval, lambda=0.49 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor hybrid --n_images -1 --lambdas 0.49

echo "########## Qwen14B test complete ##########"
