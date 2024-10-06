#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

GPU_ID=5
task_type=E2E
CL=ADAPTER
bottleneck_size=500
lr=6.25e-3  # 6.25e-3
n_epochs=50  # 10
train_batch_size=10
gradient_accumulation_steps=8
seed=1
dataset_list=TM19,TM20 # SGD,TM19,TM20,MWOZ
retrain_epochs=50
retrain_lr_factor=0.1
retrain_gradient_accumulation_steps=1
#direction=1
meta_query_step=1
episodic_mem_size=200
mode=test
sample_domain=2000

# test/scorer.py
CUDA_VISIBLE_DEVICES=$GPU_ID python ./scorer.py \
  --mode $mode \
  --episodic_mem_size $episodic_mem_size \
  --task_type $task_type \
  --CL $CL \
  --bottleneck_size $bottleneck_size \
  --lr $lr \
  --n_epochs $n_epochs \
  --train_batch_size $train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --seed $seed \
  --dataset_list $dataset_list \
  --test_every_step \
  --single \
  -mi \
  --retrain \
  --retrain_lr_factor $retrain_lr_factor \
  --retrain_epochs $retrain_epochs \
  --retrain_gradient_accumulation_steps $retrain_gradient_accumulation_steps\
  -l \
  -u \
  -b \
  -aug 3 \
  -sd $sample_domain \
  -mqs $meta_query_step \
  --meta \
  --mask || exit

# -u