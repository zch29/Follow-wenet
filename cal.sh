#!/bin/bash
base=exp/zh_k2_prune_0.16B_v20260129_ctc0.1_rnnt0.9//step_431999_chunk_32_cache_5_aedw0p0_hatlw1p0_alsd0p95_aedwin1
#base=exp/zh_k2_prune_0.16B_v20260129_ctc0.1_rnnt0.9//step_110999_chunk_16_cache_5_aedw0p0_hatlw1p0_alsd0p95_aedwin1
#base=exp/zh_k2_prune_0.16B_v20260129_ctc0.1_rnnt0.9/step_92999_chunk_16_cache_5_aedw0p0_hatlw1p0_alsd0p95_aedwin1
#base=exp/zh_k2_prune_0.2B_v20260203_multimode//step_83999_chunk_16_cache_5_aedw0p0_hatlw1p0_alsd0p95_aedwin1
#base=exp/zh_k2_prune_0.2B_v20260205_stateless//step_1189999_chunk_16_cache_5_rever0/
#mode=rnnt_beam_search
mode=attention
while read -r s; do
  awk '/Overall WER/{wer=$4}/Sentence Error Rate/{ser=$4}END{print wer "/" ser}' \
    "$base/$s/$mode/wer"
done < list2


#while read line;do  echo "$(grep 'Overall WER' exp/zh_k2_prune_0.2B_v20251212_ctc0.1_rnnt0.9/step_1034999_chunk_8_cache_5/$line/rnnt_beam_search/wer | awk '{print $4}')/$(grep 'Sentence Error Rate' exp/zh_k2_prune_0.2B_v20251212_ctc0.1_rnnt0.9/step_1034999_chunk_8_cache_5/$line/rnnt_beam_search/wer | awk '{print $4}')";done < list1
