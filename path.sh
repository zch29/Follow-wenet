#!/bin/bash
#image="docker.v2.aispeech.com/hpc/ai_on_device-hao.li_k2:pytorch2.1.0-cuda11.8-v1.4.0" "docker.v2.aispeech.com/hpc/ai_on_device-from-hao.li_k2:pytorch2.1.0-cuda11.8-v1.5.0"
#export gpu_cmd="vc submit --image $image --partition pdgpu-a10 --gpu-per-task 1 --mem-per-task 6G --cpu-per-task 3 --sync"
# kaldi 包
TOOL_DIR=/hpc_stor01/project/ezdl
export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export KALDI_ROOT=$TOOL_DIR/kaldi_aispeech_2080ti #/opt/kaldi

#export LD_LIBRARY_PATH=/hpc_stor01/home/chenghao.zhao/.local/mylib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/opt/kaldi_cpu/lib:/hpc_stor01/home/jifa.cai/git_file/pytorch-idc/source/kaldi_cpu/lib:/hpc_stor01/home/yihua.zhou/.local/miniconda3/lib:$LD_LIBRARY_PATH
#[AI修改 开始位置 20260115 修正脚本相对路径依赖] 作用：无论从哪里执行脚本，都能指向当前仓库根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#[AI修改 结束位置 20260115 修正脚本相对路径依赖]
export PATH=$SCRIPT_DIR:$SCRIPT_DIR/local:$SCRIPT_DIR/utils:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet0bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/onlinebin/:$KALDI_ROOT/src/ivectorbin/:$PATH

export LC_ALL=C
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/hpc_stor01/public/modules/packages/cuda/cuda-12.2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

#[AI修改 开始位置 20260115 增加Whitening日志开关] 作用：默认关闭 Whitening 训练监控日志（需要时可在外部覆盖为 1）
: "${ICEFALL_LOG_WHITENING:=0}"
export ICEFALL_LOG_WHITENING
#[AI修改 结束位置 20260115 增加Whitening日志开关]

export PATH=$KALDI_ROOT/tools/openfst/bin:$PATH
#[AI修改 开始位置 2026-02-11 修改原因] 兼容 set -u：LD_LIBRARY_PATH 未定义时使用空串
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.1/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/hpc_stor01/home/jifa.cai/.local/lib/python3.10/libso:${LD_LIBRARY_PATH:-}
#[AI修改 结束位置 2026-02-11 修改原因] 兼容 set -u：LD_LIBRARY_PATH 未定义时使用空串
#export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.1/lib:$KALDI_ROOT/tools/OpenBLAS:$LD_LIBRARY_PATH

MYDIR=/hpc_stor01/group/on_device/gtools
#[AI修改 开始位置 20260115 固定PYTHONPATH优先级] 作用：优先使用仓库自带 icefall，避免被 conda 的 icefall 覆盖
#[AI修改 开始位置 2026-02-11 修改原因] 兼容 set -u：PYTHONPATH 未定义时使用空串
export PYTHONPATH=$SCRIPT_DIR/icefall:$SCRIPT_DIR:/hpc_stor01/home/chenghao.zhao/work25/wenet-zipformer:/hpc_stor01/home/chenghao.zhao/src/pytorch-asr-clas:$MYDIR/audiomentations:$MYDIR/text-to-speech:$SCRIPT_DIR/ap_kws/train:${PYTHONPATH:-}
#[AI修改 结束位置 2026-02-11 修改原因] 兼容 set -u：PYTHONPATH 未定义时使用空串
#[AI修改 结束位置 20260115 固定PYTHONPATH优先级]
#[AI修改 开始位置 2026-02-11 修改原因] 兼容 set -u：PYTHONPATH 未定义时使用空串
export PYTHONPATH=${PYTHONPATH:-}:/hpc_stor01/home/jifa.cai/.local/lib/python3.10/site-packages:/hpc_stor01/home/chenghao.zhao/work25/wenet-shengteng-lixu/depend/whisper
#[AI修改 结束位置 2026-02-11 修改原因] 兼容 set -u：PYTHONPATH 未定义时使用空串
#export PYTHONPATH=$PYTHONPATH:/hpc_stor01/home/jifa.cai/.local/lib/python3.10/site-packages:/hpc_stor02/home/jifa.cai/tools/whisper
