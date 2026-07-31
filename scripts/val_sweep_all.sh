#!/bin/bash
#SBATCH --job-name=val_sweep
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/val_sweep_%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/val_sweep_%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate
cd /user/gaad2403/Applied
PYTHON=/fs/dss/home/gaad2403/negation_env/bin/python

MCQ_VAL=outputs/negbench/COCO_val_mcq_val.csv
RETRIEVAL_VAL=outputs/negbench/COCO_val_negated_retrieval_val.csv
COCO_ROOT=outputs/coco/images/val2017
LLAMA_CFG=configs/stage2.yaml
QWEN_CFG=configs/stage2_qwen.yaml

echo "########## 1) rule x MCQ ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $LLAMA_CFG --mcq_csv $MCQ_VAL --coco_root $COCO_ROOT \
  --extractor rule --n_rows -1 \
  --lambdas 0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90

echo "########## 2) rule x Retrieval ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $LLAMA_CFG --retrieval_csv $RETRIEVAL_VAL --coco_root $COCO_ROOT \
  --extractor rule --n_images -1 \
  --lambdas 0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60

echo "########## 3) Llama-llm x MCQ ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $LLAMA_CFG --mcq_csv $MCQ_VAL --coco_root $COCO_ROOT \
  --extractor llm --n_rows -1 \
  --lambdas 0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90

echo "########## 4) Llama-hybrid x MCQ ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $LLAMA_CFG --mcq_csv $MCQ_VAL --coco_root $COCO_ROOT \
  --extractor hybrid --n_rows -1 \
  --lambdas 0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10

echo "########## 5) Llama-llm x Retrieval ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $LLAMA_CFG --retrieval_csv $RETRIEVAL_VAL --coco_root $COCO_ROOT \
  --extractor llm --n_images -1 \
  --lambdas 0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70

echo "########## 6) Llama-hybrid x Retrieval ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $LLAMA_CFG --retrieval_csv $RETRIEVAL_VAL --coco_root $COCO_ROOT \
  --extractor hybrid --n_images -1 \
  --lambdas 0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65

echo "########## 7) Qwen-llm x MCQ ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $QWEN_CFG --mcq_csv $MCQ_VAL --coco_root $COCO_ROOT \
  --extractor llm --n_rows -1 \
  --lambdas 0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95

echo "########## 8) Qwen-hybrid x MCQ ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $QWEN_CFG --mcq_csv $MCQ_VAL --coco_root $COCO_ROOT \
  --extractor hybrid --n_rows -1 \
  --lambdas 0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05

echo "########## 9) Qwen-llm x Retrieval ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $QWEN_CFG --retrieval_csv $RETRIEVAL_VAL --coco_root $COCO_ROOT \
  --extractor llm --n_images -1 \
  --lambdas 0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65

echo "########## 10) Qwen-hybrid x Retrieval ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $QWEN_CFG --retrieval_csv $RETRIEVAL_VAL --coco_root $COCO_ROOT \
  --extractor hybrid --n_images -1 \
  --lambdas 0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65

echo "########## All val sweeps complete ##########"
