bash icefall/egs/multi_zh_en/ASR/zipformer/extend_chenghao/vc_submit_wenet_jsonl_to_cuts.sh \
  --nj 64 --stage 1 --stop_stage 3 \
  /mnt/lustre/hpc_stor01/home/chenghao.zhao/work25/project/data/fromd4/tr/data.jsonl \
  /mnt/lustre/hpc_stor01/home/chenghao.zhao/work25/project/data/fromd4/cv/data.jsonl \
  /mnt/lustre/hpc_stor01/home/chenghao.zhao/work25/project/data/icefall_data
