#!/bin/bash
#SBATCH --job-name=stage2_eval_all
#SBATCH --partition=mpcg.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/user/gaad2403/Applied/logs/stage2_eval_all_%j.log
#SBATCH --error=/user/gaad2403/Applied/logs/stage2_eval_all_%j.err

source /fs/dss/home/gaad2403/negation_env/bin/activate
cd /user/gaad2403/Applied

PYTHON=/fs/dss/home/gaad2403/negation_env/bin/python
CONFIG=configs/stage2.yaml
PROCRUSTES_DIR=outputs/vectors/stage2_llama3.1-8b_clipvitb32_ccneg_20260727_041404
NATIVE_VECTOR=$PROCRUSTES_DIR/clip_neg_dir_native.npy
TRANSFERRED_VECTOR=$PROCRUSTES_DIR/clip_neg_dir_transferred.npy
FINAL_VECTOR=outputs/vectors/stage2_vector_finetune_20260727_112810/clip_dir_finetuned.npy
ALPHA=0.3
LAM=1.9
COCO_ROOT=outputs/coco/images/val2017
MCQ_CSV=outputs/negbench/COCO_val_mcq_llama3.1_rephrased.csv
RETRIEVAL_CSV=outputs/negbench/COCO_val_retrieval.csv
NEG_RETRIEVAL_CSV=outputs/negbench/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv

# content-aware 평가는 캡션마다 LLM 생성을 돌려야 해서 훨씬 느려요.
# 처음엔 -1(전체)로 두되, 시간이 부족하면 아래 두 값을 줄여서(예: 1000, 500) 먼저 확인하세요.
CA_N_ROWS=-1
CA_N_IMAGES=-1

echo "########## 1) Baseline: NegCLIP on CC-Neg eval set ##########"
$PYTHON -u src/stage2/run_baseline_eval.py \
  --config $CONFIG --model negclip

echo "########## 2) NegBench MCQ, uncorrected (CLIP baseline) ##########"
$PYTHON -u src/stage2/run_negbench_mcq_eval.py \
  --config $CONFIG --mcq_csv $MCQ_CSV --coco_root $COCO_ROOT

echo "########## 3) NegBench MCQ, native direction ##########"
$PYTHON -u src/stage2/run_negbench_mcq_eval.py \
  --config $CONFIG --mcq_csv $MCQ_CSV --coco_root $COCO_ROOT \
  --vector_path $NATIVE_VECTOR --alpha $ALPHA

echo "########## 4) NegBench MCQ, transferred direction (Procrustes only) ##########"
$PYTHON -u src/stage2/run_negbench_mcq_eval.py \
  --config $CONFIG --mcq_csv $MCQ_CSV --coco_root $COCO_ROOT \
  --vector_path $TRANSFERRED_VECTOR --alpha $ALPHA

echo "########## 5) NegBench MCQ, final vector (ours, fine-tuned) ##########"
$PYTHON -u src/stage2/run_negbench_mcq_eval.py \
  --config $CONFIG --mcq_csv $MCQ_CSV --coco_root $COCO_ROOT \
  --vector_path $FINAL_VECTOR --alpha $ALPHA

echo "########## 6) NegBench MCQ, content-aware (LLM extraction + Eq.2) ##########"
$PYTHON -u src/stage2/run_negbench_mcq_content_aware.py \
  --config $CONFIG --mcq_csv $MCQ_CSV --coco_root $COCO_ROOT \
  --lam $LAM --n_rows $CA_N_ROWS

echo "########## 7) NegBench Retrieval, affirmative (sanity check) ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_eval.py \
  --config $CONFIG --retrieval_csv $RETRIEVAL_CSV --coco_root $COCO_ROOT

echo "########## 8) NegBench Retrieval, negated, uncorrected ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_eval.py \
  --config $CONFIG --retrieval_csv $NEG_RETRIEVAL_CSV --coco_root $COCO_ROOT

echo "########## 9) NegBench Retrieval, negated, native direction ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_eval.py \
  --config $CONFIG --retrieval_csv $NEG_RETRIEVAL_CSV --coco_root $COCO_ROOT \
  --vector_path $NATIVE_VECTOR --alpha $ALPHA

echo "########## 10) NegBench Retrieval, negated, transferred direction ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_eval.py \
  --config $CONFIG --retrieval_csv $NEG_RETRIEVAL_CSV --coco_root $COCO_ROOT \
  --vector_path $TRANSFERRED_VECTOR --alpha $ALPHA

echo "########## 11) NegBench Retrieval, negated, final vector (ours, fine-tuned) ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_eval.py \
  --config $CONFIG --retrieval_csv $NEG_RETRIEVAL_CSV --coco_root $COCO_ROOT \
  --vector_path $FINAL_VECTOR --alpha $ALPHA

echo "########## 12) NegBench Retrieval, negated, content-aware (LLM extraction + Eq.2) ##########"
$PYTHON -u src/stage2/run_negbench_retrieval_content_aware.py \
  --config $CONFIG --retrieval_csv $NEG_RETRIEVAL_CSV --coco_root $COCO_ROOT \
  --lam $LAM --n_images $CA_N_IMAGES

echo "########## All evaluations complete ##########"