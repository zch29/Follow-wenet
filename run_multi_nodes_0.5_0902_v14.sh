#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of npus
if command -v npu-smi info &> /dev/null; then
  num_npus=$(npu-smi info -l | grep "Total Count" | awk '{print $4}')
  npu_list=$(seq -s, 0 $((num_npus-1)))
else
  num_npus=-1
  npu_list="-1"
fi

# You can also manually specify ASCEND_RT_VISIBLE_DEVICES
# if you don't want to utilize all available NPU resources.
export ASCEND_RT_VISIBLE_DEVICES="${npu_list}"
echo "ASCEND_RT_VISIBLE_DEVICES is ${ASCEND_RT_VISIBLE_DEVICES}"
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2 

stage=4 # start from 0 if you need to start from data preparation
stop_stage=4

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
num_nodes=6
job_id=20250902

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=ark

train_set=zh_250902_v14/
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_u2++_conformer.yaml: U2++ conformer
# 6. conf/train_u2++_transformer.yaml: U2++ transformer
# 7. conf/train_u2++_conformer.yaml: U2++ lite conformer, must load a well
#    trained model, and freeze encoder module, otherwise there will be a
#    autograd error
train_config=conf/train_aicar_u2++_conformer_0.5B.yaml
dir=exp/zh_250902_v14_u2++_conformer_0.5B
tensorboard_dir=tensorboard
checkpoint=exp/zh_250902_v14_u2++_conformer_0.5B/step_1169999.pt
num_workers=4
prefetch=4

train_engine=torch_ddp

deepspeed_config=conf/ds_stage1.json
deepspeed_save_states="model_only"

# Syntax error: Bad for loop variable
. tools/parse_options.sh || exit 1;

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  mkdir -p $dir
  num_npus=$(echo $ASCEND_RT_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "hccl" for npu if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="hccl"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  echo "$0: using ${train_engine}"

  # NOTE(xcsong): Both ddp & deepspeed can be launched by torchrun
  # NOTE(xcsong): To unify single-node & multi-node training, we add
  #               all related args. You should change `nnodes` &
  #               `rdzv_endpoint` for multi-node, see
  #               https://pytorch.org/docs/stable/elastic/run.html#usage
  #               https://github.com/wenet-e2e/wenet/pull/2055#issuecomment-1766055406
  #               `rdzv_id` - A user-defined id that uniquely identifies the worker group for a job.
  #                           This id is used by each node to join as a member of a particular worker group.
  #               `rdzv_endpoint` - The rendezvous backend endpoint; usually in form <host>:<port>.
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_npus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_npus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py \
      --device "npu" \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data data_my/$train_set/train/data.jsonl \
      --cv_data data_my/$train_set/dev/data.jsonl \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

