#!/bin/bash
#SBATCH --job-name=test_final
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/test_final_%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/test_final_%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate
cd /user/gaad2403/Applied
PYTHON=/fs/dss/home/gaad2403/negation_env/bin/python

MCQ_TEST=outputs/negbench/COCO_val_mcq_test.csv
RETRIEVAL_TEST=outputs/negbench/COCO_val_negated_retrieval_test.csv
COCO_ROOT=outputs/coco/images/val2017
LLAMA_CFG=configs/stage2.yaml
QWEN_CFG=configs/stage2_qwen.yaml

echo "########## 1) rule x MCQ, lambda=0.85 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $LLAMA_CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor rule --n_rows -1 --lambdas 0.85

echo "########## 2) rule x Retrieval, lambda=0.45 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $LLAMA_CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor rule --n_images -1 --lambdas 0.45

echo "########## 3) Llama-llm x MCQ, lambda=0.81 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $LLAMA_CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor llm --n_rows -1 --lambdas 0.81

echo "########## 4) Llama-hybrid x MCQ, lambda=0.92 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $LLAMA_CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor hybrid --n_rows -1 --lambdas 0.92

echo "########## 5) Llama-llm x Retrieval, lambda=0.63 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $LLAMA_CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor llm --n_images -1 --lambdas 0.63

echo "########## 6) Llama-hybrid x Retrieval, lambda=0.47 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $LLAMA_CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor hybrid --n_images -1 --lambdas 0.47

echo "########## 7) Qwen-llm x MCQ, lambda=0.85 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $QWEN_CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor llm --n_rows -1 --lambdas 0.85

echo "########## 8) Qwen-hybrid x MCQ, lambda=1.00 ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $QWEN_CFG --mcq_csv $MCQ_TEST --coco_root $COCO_ROOT \
  --extractor hybrid --n_rows -1 --lambdas 1.00

echo "########## 9) Qwen-llm x Retrieval, lambda=0.62 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $QWEN_CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor llm --n_images -1 --lambdas 0.62

echo "########## 10) Qwen-hybrid x Retrieval, lambda=0.62 ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $QWEN_CFG --retrieval_csv $RETRIEVAL_TEST --coco_root $COCO_ROOT \
  --extractor hybrid --n_images -1 --lambdas 0.62

echo "########## All test runs complete ##########"
