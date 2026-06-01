# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.common import TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard', 'ark'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_hat_aed_alsd_beam_search
                             rnnt_hat_offline_aed_rescoring
                             rnnt_offline_aed_greedy_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    #[AI修改 开始位置 20260115 推理参数透传Trick1Trick2] 作用：Trick1+Trick2 推理侧参数（共享 predictor + ALSD blank 门控）
    parser.add_argument('--aed_weight',
                        type=float,
                        default=0.0,
                        help='AED 分数权重（0 表示关闭 AED 联合解码）')
    parser.add_argument('--alsd_blank_threshold',
                        type=float,
                        default=1.0,
                        help='ALSD blank 概率阈值（越小越啥少触发 AED）')
    parser.add_argument('--hat_label_weight',
                        type=float,
                        default=1.0,
                        help='HAT label 分数权重（0 表示仅保留 blank 约束）')
    parser.add_argument('--aed_attend_window',
                        type=int,
                        default=0,
                        help='AED attention 窗口（0 表示全前缀；1 近似单帧；建议 8~32）')
    parser.add_argument('--max_consecutive_repeats',
                        type=int,
                        default=0,
                        help='最大连续重复 token 次数（0 表示不限制；建议 2）')
    #[AI修改 结束位置 20260115 推理参数透传Trick1Trick2]
    #[AI修改 开始位置 20260112 新增HAT流式离线AED重打分解码]
    parser.add_argument('--offline_aed_weight',
                        type=float,
                        default=0.0,
                        help='离线 AED 重打分权重（0 表示关闭）')
    parser.add_argument('--offline_aed_nbest',
                        type=int,
                        default=0,
                        help='离线 AED 重打分的 nbest 数（0 表示使用 beam_size）')
    parser.add_argument('--offline_aed_max_len',
                        type=int,
                        default=0,
                        help='离线 AED 自解码最大长度（0 表示自动 = 2*T）')
    parser.add_argument('--offline_aed_eos_id',
                        type=int,
                        default=-1,
                        help='离线 AED 自解码停止符号 id（-1 表示不启用）')
    parser.add_argument('--offline_aed_attend_window',
                        type=int,
                        default=0,
                        help='离线 AED attention 窗口（0 表示全序列）')
    parser.add_argument('--offline_aed_max_consecutive_repeats',
                        type=int,
                        default=0,
                        help='离线 AED 自解码最大连续重复次数（0 表示不限制）')
    #[AI修改 结束位置 20260112 新增HAT流式离线AED重打分解码]
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')

    parser.add_argument('--use_lora',
                        type=bool,
                        default=False,
                        help='''Whether to use lora for biasing''')
    parser.add_argument("--lora_ckpt_path",
                        default=None,
                        type=str,
                        help="lora checkpoint path.")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['cycle'] = 1
    test_conf['list_shuffle'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)
    #breakpoint()
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))

    context_graph = None
    if 'decoding-graph' in args.context_bias_mode:
        context_graph = ContextGraph(args.context_list_path,
                                     tokenizer.symbol_table,
                                     configs['tokenizer_conf']['bpe_path'],
                                     args.context_graph_score)

    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))

    # TODO(Dinghao Zhou): Support RNN-T related decoding
    # TODO(Lv Xiang): Support k2 related decoding
    # TODO(Kaixun Huang): Support context graph
    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])

    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                keys = batch["keys"]
                feats = batch["feats"].to(device)
                target = batch["target"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                infos = {"tasks": batch["tasks"], "langs": batch["langs"]}
                results = model.decode(
                    args.modes,
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                    context_graph=context_graph,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty,
                    length_penalty=args.length_penalty,
                    #[AI修改 开始位置 20260115 推理参数透传Trick1Trick2] 作用：把 Trick1+Trick2 参数透传到模型 decode
                    aed_weight=args.aed_weight,
                    alsd_blank_threshold=args.alsd_blank_threshold,
                    hat_label_weight=args.hat_label_weight,
                    aed_attend_window=args.aed_attend_window,
                    max_consecutive_repeats=args.max_consecutive_repeats,
                    #[AI修改 结束位置 20260115 推理参数透传Trick1Trick2]
                    #[AI修改 开始位置 20260112 新增HAT流式离线AED重打分解码]
                    offline_aed_weight=args.offline_aed_weight,
                    offline_aed_nbest=args.offline_aed_nbest,
                    offline_aed_max_len=args.offline_aed_max_len,
                    offline_aed_eos_id=args.offline_aed_eos_id,
                    offline_aed_attend_window=args.offline_aed_attend_window,
                    offline_aed_max_consecutive_repeats=args.offline_aed_max_consecutive_repeats,
                    #[AI修改 结束位置 20260112 新增HAT流式离线AED重打分解码]
                    infos=infos)
                #breakpoint()
                for i, key in enumerate(keys):
                    for mode, hyps in results.items():
                        tokens = hyps[i].tokens
                        line = '{} {}'.format(key,
                                              tokenizer.detokenize(tokens)[0])
                        logging.info('{} {}'.format(mode.ljust(max_format_len),
                                                    line))
                        files[mode].write(line + '\n')
        for mode, f in files.items():
            f.close()


if __name__ == '__main__':
    main()
