#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

GPU_ID=7
task_type=E2E
CL=ADAPTER
bottleneck_size=900
lr=3e-3  # 6.25e-3
n_epochs=50  # 10
train_batch_size=10
gradient_accumulation_steps=8
seed=2
dataset_list=TM19,TM20 # SGD,TM19,TM20,MWOZ
retrain_epochs=3
retrain_lr_factor=0.1
retrain_gradient_accumulation_steps=1
#direction=1
meta_query_step=1
episodic_mem_size=200
sample_domain=2000

# train_adapter_expand train_wo_pytorch_lightning
#CUDA_VISIBLE_DEVICES=$GPU_ID python ./train_adapter_expand.py \
#  --episodic_mem_size $episodic_mem_size \
#  --task_type $task_type \
#  --CL $CL \
#  --bottleneck_size $bottleneck_size \
#  --lr $lr \
#  --n_epochs $n_epochs \
#  --train_batch_size $train_batch_size \
#  --gradient_accumulation_steps $gradient_accumulation_steps \
#  --seed $seed \
#  --dataset_list $dataset_list \
#  --test_every_step \
#  --single \
#  -mi \
#  -e \
#  -nm 3 \
#  \
#  -aug 3 \
#  -sd $sample_domain \
#  -mqs $meta_query_step \
#  \
#  --mask || exit

#CUDA_VISIBLE_DEVICES=$GPU_ID python ./train_adapter_expand.py \
#  --episodic_mem_size $episodic_mem_size \
#  --task_type $task_type \
#  --CL $CL \
#  --bottleneck_size $bottleneck_size \
#  --lr $lr \
#  --n_epochs $n_epochs \
#  --train_batch_size $train_batch_size \
#  --gradient_accumulation_steps $gradient_accumulation_steps \
#  --seed $seed \
#  --dataset_list $dataset_list \
#  --test_every_step \
#  --single \
#  -mi \
#  --retrain \
#  --retrain_lr_factor $retrain_lr_factor \
#  --retrain_epochs $retrain_epochs \
#  --retrain_gradient_accumulation_steps $retrain_gradient_accumulation_steps\
#  -k1 \
#  \
#  -b \
#  \
#  -sd $sample_domain \
#  -mqs $meta_query_step \
#  --meta \
#  -sm \
#  -na 3 \
#  \
#  --mask || exit

#-u
#-aug 3

#  -t 5
#-nm 3

# -md
# -l -v
# -de $direction
# -f -mc -task 1

# --test_every_step -u
#  --meta --retrain -t1 -cm
# --retrain_lr_factor 1
#python ./scorer.py \

#mode=test
# scorer.py train_wo_pytorch_lightning
# --mode $mode
# CUDA_VISIBLE_DEVICES=$GPU_ID python ./scorer.py \
#   --mode test\
#   --episodic_mem_size $episodic_mem_size \
#   --task_type $task_type \
#   --CL $CL \
#   --bottleneck_size $bottleneck_size \
#   --lr $lr \
#   --n_epochs $n_epochs \
#   --train_batch_size $train_batch_size \
#   --gradient_accumulation_steps $gradient_accumulation_steps \
#   --seed $seed \
#   --dataset_list $dataset_list \
#   --test_every_step \
#   --single \
#   -mi \
#   --retrain \
#   --retrain_lr_factor $retrain_lr_factor \
#   --retrain_epochs $retrain_epochs \
#   --retrain_gradient_accumulation_steps $retrain_gradient_accumulation_steps\
#   -k1 \
#   -u \
#   -b \
#   -aug 3 \
#   -sd $sample_domain \
#   -mqs $meta_query_step \
#   --meta \
#   -sm \
#   -na 3 \
#   --mask || exit

CUDA_VISIBLE_DEVICES=$GPU_ID python ./test_fwt.py \
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
  -k1 \
  -u \
  -b \
  -aug 3 \
  -sd $sample_domain \
  -mqs $meta_query_step \
  --meta \
  -sm \
  -na 3 \
  --mode test \
  -fwt \
  -notil \
  --mask || exit