#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
#. ./path.sh || exit 1;

# Configuration parameters
run_mode=hpc
data_type=ark

dir=exp/zh_k2_prune_0.2B
decode_checkpoint=$dir/step_299999.pt
#data="/hpc_stor01/home/chenghao.zhao/work25/wenet-shengteng-lixu/testsets/" #"/hpc_stor01/home/chenghao.zhao/work23/online_e2e_clas/utils/wbs_nbest/newmulticase/example/testscp/"
data="/mnt/lustre/hpc_stor01/home/chenghao.zhao/work25/wenet-shengteng-lixu/testsets_add"

# Define test sets array
test_sets=(
#"aicar_real_LLMv1_250220"
"aicar_real_LLMv1_250220"
"aicar_real_LLMv2_250220"
"baike_tts"
"BanMa.Car_real_2023Q1"
"Biyadi.Car_real_2024Q3"
"Biyadi.Car_real_2024Q4"
"ChangCheng.Car_real_2024Q3"
"ChangCheng.Car_real_2024Q4"
"gpt4"
"JiLi.Car_real_2024Q3"
"JiLi.Car_real_2024Q4"
"jili_hx_audiobook_250318"
"jili_hx_CBVR_250318"
"jili_hx_chat_250318"
"jili_hx_chewai_250318"
"jili_hx_cmd_250318"
"jili_hx_dongche_250318"
"jili_hx_geely_250318"
"jili_hx_media_250318"
"jili_hx_music_250318"
"jili_hx_navi_250318"
"jili_hx_POI_250318"
"jili_hx_selfdriving_250318"
"jili_hx_soundgen_250318"
"jili_hx_video_250318"
"jili_hx_weather_250318"
"jili_hx_web_250318"
"wuling_LLM_ambient_241231"
"wuling_LLM_attraction_241231"
"wuling_LLM_chat_241231"
"wuling_LLM_draw_241231"
"wuling_LLM_food_241231"
"wuling_LLM_music_241231"
"wuling_LLM_schedule_241231"
"wuling_LLM_sensitive_241231"
"wuling_LLM_websearch_241231"
"jili_gen_navi_250320"
"jili_gen_CBVR_250320"
"jili_gen_POI_250320"
"jili_gen_chewai_250320"
"jili_gen_selfdriving_250320"
"jili_gen_media_250320"
"jili_gen_cmd_250320"
"jili_gen_audiobook_250320"
"jili_gen_dongche_250320"
"jili_gen_web_250320"
"jili_gen_video_250320"
"jili_gen_music_250320"
"jili_gen_chat_250320"
"jili_gen_geely_250320"
"jili_gen_soundgen_250320"
"jili_gen_weather_250320"
"jike_chat_250228"
"jike_baike_250228"
"jike_baike_250311"
"acar_byd_cen_250325"
"acar_byd_en_250325"
"acar_byd_jiwai_250325"
)

decode_modes="rnnt_beam_search"
beam_size=8
batch_size=1
decoding_chunk_size=8
ctc_weight=0.15
reverse_weight=0.3

# Define the Docker image and vc submit commands
image="docker.v2.aispeech.com/hpc/ai_on_device-hao.li_k2:pytorch2.1.0-cuda11.8-v1.4.0"
cpu_cmd="vc submit --image $image --partition pdcpu --mem-per-task 36G --cpu-per-task 4"
cuda_cmd="vc submit --image $image --partition pdgpu-3090 --mem-per-task 16G --cpu-per-task 8 --gpu-per-task 1"

# Process flow control
stage=0
stop_stage=0

set -e
set -u
set -o pipefail

. tools/parse_options.sh || exit 1;

# Create results directory
result_dir=$dir/inference_streaming_chunk16
mkdir -p $result_dir/log

simulate_streaming=true
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: Running inference with vc submit in ${run_mode} mode"
  
  for test_set in "${test_sets[@]}"; do
    test_dir=$data/$test_set
    echo "Processing test set: $test_set"
    
    if [ "$run_mode" == "local" ]; then
      # Run locally without vc submit
      python wenet/bin/recognize.py --gpu -1 \
        --device "cpu" \
        --modes $decode_modes \
        --config $dir/step_39999.yaml \
        --data_type $data_type \
        --test_data $test_dir/data.jsonl \
        --checkpoint $decode_checkpoint \
        --beam_size $beam_size \
        --batch_size $batch_size \
        --ctc_weight $ctc_weight \
        --reverse_weight $reverse_weight \
        --result_dir $result_dir/$test_set \
	--num_decoding_left_chunks 5 \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
        ${simulate_streaming:+--simulate_streaming} 
    elif [ "$run_mode" == "hpc" ]; then
      # Run with vc submit on HPC
      log_dir=$result_dir/$test_set/log
      mkdir -p $log_dir
       
      # Submit the job using vc submit
      $cpu_cmd -j "wenet-decode" -pj "none" JOB=1:1 $log_dir/decode.JOB.log --cmd \
        "python wenet/bin/recognize.py \
          --device \"cpu\" \
          --modes $decode_modes \
          --config $dir/step_299999.yaml \
          --data_type $data_type \
          --test_data $test_dir/data.jsonl \
          --checkpoint $decode_checkpoint \
          --beam_size $beam_size \
          --batch_size $batch_size \
          --ctc_weight $ctc_weight \
          --reverse_weight $reverse_weight \
          --result_dir $result_dir/$test_set/ \
	  --num_decoding_left_chunks 5 \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
	  ${simulate_streaming:+--simulate_streaming}" \
        2>&1 | tee $log_dir/log.JOB &
    else
      echo "Unknown mode: $run_mode"
      exit 1
    fi
  done
  # Wait for all background jobs to complete
  wait
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Computing WER"
  
  for test_set in "${test_sets[@]}"; do
    test_dir=$data/$test_set
    echo "Computing WER for test set: $test_set"
    
    for mode in ${decode_modes}; do
      mkdir -p $result_dir/$test_set/
      if [ -f "$test_dir/text" ]; then
        TEXT="$test_dir/text"
      else
        TEXT="$test_dir/use.text"
      fi
      module add anaconda/3
      python /hpc_stor01/home/chenghao.zhao/work25/wenet-shengteng-lixu/extend_zch//compute-wer-yuchen.py --char=1 --v=1 \
        $TEXT $result_dir/$test_set/$mode/text > $result_dir/$test_set/$mode/wer
    done
  done
fi

