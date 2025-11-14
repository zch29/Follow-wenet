#!/bin/bash
#image="docker.v2.aispeech.com/hpc/ai_on_device-hao.li_k2:pytorch2.1.0-cuda11.8-v1.4.0"
#export gpu_cmd="vc submit --image $image --partition pdgpu-a10 --gpu-per-task 1 --mem-per-task 6G --cpu-per-task 3 --sync"
# kaldi 包
TOOL_DIR=/hpc_stor01/project/ezdl
export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export KALDI_ROOT=$TOOL_DIR/kaldi_aispeech_2080ti #/opt/kaldi

#export LD_LIBRARY_PATH=/hpc_stor01/home/chenghao.zhao/.local/mylib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/opt/kaldi_cpu/lib:/hpc_stor01/home/jifa.cai/git_file/pytorch-idc/source/kaldi_cpu/lib:/hpc_stor01/home/yihua.zhou/.local/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=$PWD:$PWD/local:$PWD/utils:$SCRIPT_DIR:$SCRIPT_DIR/local:$SCRIPT_DIR/utils/:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet0bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/onlinebin/:$KALDI_ROOT/src/ivectorbin/:$PATH

export LC_ALL=C
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME=/hpc_stor01/public/modules/packages/cuda/cuda-12.2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PATH=$KALDI_ROOT/tools/openfst/bin:$PATH
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/hpc_stor01/home/jifa.cai/.local/lib/python3.10/libso:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst-1.6.1/lib:$KALDI_ROOT/tools/OpenBLAS:$LD_LIBRARY_PATH

MYDIR=/hpc_stor01/group/on_device/gtools
export PYTHONPATH=/hpc_stor01/home/chenghao.zhao/work25/wenet-shengteng-lixu:/hpc_stor01/home/chenghao.zhao/src/pytorch-asr-clas:$MYDIR/audiomentations:$MYDIR/text-to-speech:$PWD/ap_kws/train:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/hpc_stor01/home/jifa.cai/.local/lib/python3.10/site-packages:/hpc_stor01/home/chenghao.zhao/work25/wenet-shengteng-lixu/depend/whisper
#export PYTHONPATH=$PYTHONPATH:/hpc_stor01/home/jifa.cai/.local/lib/python3.10/site-packages:/hpc_stor02/home/jifa.cai/tools/whisper
