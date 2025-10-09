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
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2025

# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!

#data=/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/origin
#data_url=www.openslr.org/resources/33

#nj=16
#dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
#data_type=raw
data_type=ark
#num_utts_per_shard=1000

train_set=zh_debug
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
#train_config=conf/train_aicar_u2++_conformer_0.2B.yaml 
#train_config=conf/conformer_rnnt.yaml
#train_config=conf/conformer_rnnt_small.yaml
train_config=conf/conformer_rnnt_small_rnnt+ctc+att.yaml
dir=exp/conformer_debug_v1
tensorboard_dir=tensorboard
checkpoint=
num_workers=1
prefetch=2

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

# specify your distributed training method among ['torch_ddp', 'torch_fsdp', 'deepspeed']
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
  num_npus=1
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_npus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py \
      --device "npu" \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data data/$train_set/train/data.jsonl \
      --cv_data data/$train_set/dev/data.jsonl \
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.3
  reverse_weight=0.5
  python wenet/bin/recognize.py \
    --device "npu" \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data data/test/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 32 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $dir/$mode/text > $dir/$mode/wer
  done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi
