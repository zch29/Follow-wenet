#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;


# Process flow control
stage=1
stop_stage=1

# Configuration parameters
run_mode=hpc
data_type=ark

#dir=exp/zh_k2_prune_0.1B_v20260104_multimode
#dir=exp/zh_k2_prune_0.1B_v20260108_multimode/
dir=exp/zh_k2_prune_0.16B_v20260129_ctc0.1_rnnt0.9/
decode_checkpoint=step_431999.pt
decode_yaml=step_431999.yaml
decode_nj=10
#data="/hpc_stor01/home/yuchen.yan/codes/wenet-shengteng-lixu/testsets"
data="/mnt/lustre/hpc_stor01/home/xumao.wu/test_folder/aicar_llm/test_sets"

false && test_sets=(
"ptp_aicar_llm_comm"
"ptp_aicar_llm_music"
"ptp_aicar_llm_navi"
"ptp_aicar_llm_phone"
"ptp_aicar_llm_radio"
)
false && test_sets=(
"jili_hx_weather_250318"
)
test_sets=(
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
)

false && test_sets=(
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
"acar_benz_poi_240709"
"acar_benz_singer_241211"
"acar_benz_song_241211"
"aicar_byd_xf_v1_250402"
"aicar_byd_xf_v2_250402"
"aicar_byd_xf_v3_250402"
"aicar_byd_xf_chat_250407"
"acar_F5tts_zh_en_250213"
"all_byd_text"
"all_car_ner"
"all_english_business"
"all_english_name"
"all_english_new"
"all_top3500_english_word"
"benz_zhejiang"
"benz_shanghai"
"benz_guangdong"
"benz_fujian"
"benz_anhui"
"BYD_cen_0327"
"tengshiHT_3136_en"
"tengshiHT_3136_cen"
"byd_yw_chat_250421"
"byd_yw_domain_250421"
"chn_furnitureasr_250415"
"ptp_aicar_llm_comm"
"ptp_aicar_llm_music"
"ptp_aicar_llm_navi"
"ptp_aicar_llm_phone"
"ptp_aicar_llm_radio"
"byd_ptp_llm_test_0508"
)

#decode_modes="rnnt_hat_aed_alsd_beam_search"
#decode_modes="rnnt_hat_offline_aed_rescoring" #"rnnt_hat_aed_alsd_beam_search" #"rnnt_beam_search "
decode_modes="attention"
beam_size=10
batch_size=1
ctc_weight=0.1
reverse_weight=0
decoding_chunk_size=32
num_decoding_left_chunks=5
#[开始位置 20260115 解码参数可配置与输出目录tag] 作用：支持 HAT+AED 融合/门控参数可配置，并把参数写入输出目录避免覆盖
aed_weight=0.0
hat_label_weight=1.0
alsd_blank_threshold=0.95
aed_attend_window=1
max_consecutive_repeats=2
decode_tag=""
#[结束位置 20260115 解码参数可配置与输出目录tag]

#[开始位置 20260112 新增离线AED解码参数]
# 作用：支持离线 AED 重打分/自解码的参数透传；默认关闭，不影响现有解码
offline_aed_weight=0
offline_aed_nbest=10
offline_aed_max_len=0
offline_aed_eos_id=-1
offline_aed_attend_window=0
offline_aed_max_consecutive_repeats=0
#[结束位置 20260112 新增离线AED解码参数]
# Define the Docker image and vc submit commands
image="docker.v2.aispeech.com/hpc/ai_on_device-hao.li_k2:pytorch2.1.0-cuda11.8-v1.4.0"
cpu_cmd="vc submit --image $image --partition pdcpu --mem-per-task 24G --cpu-per-task 2 "
cuda_cmd="vc submit --image $image --partition pdgpu-3090 --mem-per-task 16G --cpu-per-task 8 --gpu-per-task 1"

# Process flow control
#stage=0
#stop_stage=0

set -e
set -u
set -o pipefail

. tools/parse_options.sh || exit 1;

# Create results directory
#[开始位置 20260115 输出目录包含关键参数] 作用：输出目录包含关键参数；若已存在则移动到可删除目录避免覆盖
tag_aed_weight=${aed_weight//./p}
tag_hat_label_weight=${hat_label_weight//./p}
tag_alsd_blank_threshold=${alsd_blank_threshold//./p}
run_tag=${decode_tag:-aedw${tag_aed_weight}_hatlw${tag_hat_label_weight}_alsd${tag_alsd_blank_threshold}_aedwin${aed_attend_window}}
result_dir=$dir/$(basename $decode_checkpoint .pt)_chunk_${decoding_chunk_size}_cache_${num_decoding_left_chunks}_${run_tag}
# if [ -e "$result_dir" ]; then
#   ts=$(date +%Y%m%d_%H%M%S)
#   mv "$result_dir" "${result_dir}_can_delete_${ts}"
# fi
#[结束位置 20260115 输出目录包含关键参数]

simulate_streaming=true
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Stage 0: Running inference with vc submit in ${run_mode} mode"

  for test_set in "${test_sets[@]}"; do
    echo "Processing test set: $test_set"
    log_dir=$result_dir/$test_set/log
    mkdir -p $log_dir
    test_dir=$data/$test_set
    /mnt/lustre/hpc_stor01/home/yuchen.yan/tools/miniforge3/bin/python split_lines.py $test_dir/data.jsonl $decode_nj $log_dir
    mkdir -p $log_dir
      # Submit the job using vc submit
    #[开始位置 20260115 解码参数透传recognize] 作用：把 HAT+AED 融合/门控参数透传到 recognize.py
    $cpu_cmd -j "wenet-decode" -pj "none" JOB=1:$decode_nj $log_dir/decode.JOB.log --cmd \
        "python wenet/bin/recognize.py \
          --device \"cpu\" \
          --modes $decode_modes \
          --config $dir/$decode_yaml \
          --data_type $data_type \
          --test_data $log_dir/split_JOB.txt \
          --checkpoint $dir/$decode_checkpoint \
          --beam_size $beam_size \
          --batch_size $batch_size \
          --ctc_weight $ctc_weight \
          --result_dir $result_dir/$test_set/log/JOB \
	        --num_decoding_left_chunks $num_decoding_left_chunks \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
	        ${simulate_streaming:+--simulate_streaming}" \
        2>&1 | tee $log_dir/log.JOB &
        touch $log_dir/.done
  done
  wait
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: 合并 text 后执行“档”->“挡”替换，再计算 WER"

  for test_set in "${test_sets[@]}"; do
    log_dir=$result_dir/$test_set/log
    echo "Computing WER for test set: $test_set"
    for mode in ${decode_modes}; do
      mkdir -p $result_dir/$test_set/$mode
      merged_text=$result_dir/$test_set/$mode/text

      for i in $(seq 1 $decode_nj); do
          cat $log_dir/$i/$mode/text >> "$merged_text"
      done

      if [ -f "$merged_text" ]; then
        echo "对合并后的文件执行“档”->“挡”替换: $merged_text"
        bak="${merged_text}_bak_can_delete"
        cp "$merged_text" "$bak"
        if awk '{
          key = $1;
          $1 = "";
          content = $0;
          sub(/^[[:space:]]+/, "", content);
          gsub("档", "挡", content);
          print key " " content;
        }' "$merged_text" > "${merged_text}.tmp"; then
          mv "${merged_text}.tmp" "$merged_text"
        else
          echo "替换过程中出错，开始回滚: $merged_text"
          mv "$bak" "$merged_text"
          exit 1
        fi
      else
        echo "警告: 未找到合并后的 text 文件: $merged_text"
      fi

      if [ -f "$data/$test_set/text" ]; then
          TEXT="$data/$test_set/text"
      else
          TEXT="$data/$test_set/use.text"
      fi
      /mnt/lustre/hpc_stor01/home/xumao.wu/tools/anaconda3/bin/python3.7 tools/compute-wer-v2.py --char=1 --v=1 \
        $TEXT "$merged_text" > $result_dir/$test_set/$mode/wer
    done
  done
fi
