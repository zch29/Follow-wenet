# Follow-Wenet

This repository is a research-oriented WeNet follow project focused on improving
RNN-T/Transducer support and exploring Zipformer-based streaming ASR inside the
WeNet training and decoding stack.

The original WeNet Transducer path is relatively limited for the experiments
needed here, so this branch adds a larger RNN-T/HAT experimental surface,
connects k2/icefall Zipformer components, and implements an All-in-One style
multi-mode ASR scheme that shares the encoder, predictor, and joiner across
HAT/RNN-T, AED, CTC, and internal-LM behavior.

## Main Work

- RNN-T/HAT training path
  - Expanded `wenet/transducer/transducer.py` beyond the stock Transducer path.
  - Added stable loss dict returns for standard `transducer_joint` training.
  - Added optional CTC and attention loss mixing for hybrid experiments.
  - Added k2 smoothed/pruned RNN-T loss support to avoid materializing full
    `[B, T, U, V]` tensors in large HAT experiments.

- All-in-One / multi-mode ASR experiments
  - Added `wenet/transducer/joint_optimized.py`.
  - Implemented a `MultiModeJoiner` with four modes:
    - `hat`: frame-synchronous HAT/RNN-T log probabilities.
    - `aed`: softmax cross-attention label prediction.
    - `ctc`: CTC-style output from encoder-only behavior.
    - `lm`: predictor-only internal language-model behavior.
  - Added mode registration in `wenet/utils/init_model.py` through
    `multi_mode_joiner` and `multi_mode_joint`.
  - Added gradient checkpointing and pruned HAT forward support for memory
    control.

- k2 / icefall Zipformer integration
  - Added `wenet/transformer/zipformer2_encoder.py` as a WeNet-compatible
    wrapper around icefall Zipformer2.
  - Registered `zipformer2` as a WeNet encoder.
  - Added stricter module loading to avoid importing the wrong `zipformer.py` or
    `subsampling.py` when multiple projects are on `PYTHONPATH`.
  - Added simulated streaming support using Zipformer streaming states and
    subsampling caches.

- Decoding experiments
  - Added RNN-T/HAT decoding modes in `wenet/bin/recognize.py`:
    - `rnnt_beam_search`
    - `rnnt_hat_aed_alsd_beam_search`
    - `rnnt_hat_offline_aed_rescoring`
    - `rnnt_offline_aed_greedy_search`
  - Added HAT + AED joint decoding controls, including AED score weight,
    ALSD-style blank gating, HAT label weighting, AED attention windows, and
    repeat suppression.
  - Added offline AED rescoring and greedy AED decoding from cached encoder
    output.

- Data and workflow utilities
  - Added `scripts/decode_from_kaldi_dir.py` to decode a Kaldi-style directory
    containing `text` and `wav.scp`.
  - The utility converts Kaldi metadata to WeNet raw JSONL, splits jobs, runs
    `recognize.py`, merges hypotheses, writes comparison summaries, and can
    submit decode jobs through `vc`.
  - Added unit tests under `tests/test_decode_from_kaldi_dir.py`.

## Repository Layout

```text
wenet/
  bin/                       Training and recognition entrypoints
  transducer/                RNN-T/HAT models, search, and multi-mode joiner
  transformer/               WeNet encoders plus Zipformer2 adapter
  utils/                     Model initialization, checkpoints, training utils
scripts/
  decode_from_kaldi_dir.py   Kaldi-style directory decode helper
docs/
  plans/                     Design notes for added utilities
examples/
  conf/                      Snapshot configs for RNNT, multi-mode, Zipformer
tests/
  test_decode_from_kaldi_dir.py
run_*.sh                    Example training launch scripts
decode_conformer_ark*.sh    Example decoding launch scripts
path.sh                     Local HPC/Kaldi/Python environment setup
```

Large experiment outputs, checkpoints, TensorBoard logs, data directories, and
external symlinked projects are intentionally not tracked.

The files in `examples/conf/` are copied snapshots from the local experiment
`conf` symlink so the key model structures can be inspected without the private
working tree.

## Important External Dependencies

This project is not a clean pip-install package. It is a working research tree
that expects the same style of environment as the original WeNet experiments:

- PyTorch and torchaudio.
- WeNet runtime dependencies.
- k2 when `enable_k2: true` is used.
- An icefall checkout for Zipformer2 experiments.
- Kaldi tools when using ark/Kaldi-style decode workflows.
- Internal HPC/`vc` tooling for the provided multi-node launch scripts.

For Zipformer2, set one of the following if the local `icefall` symlink is not
available:

```bash
export WENET_ICEFALL_ZIPFORMER_DIR=/path/to/icefall/egs/multi_zh_en/ASR/zipformer
```

## Typical Usage

Train with one of the example launch scripts after adapting local data, config,
and cluster paths:

```bash
bash run_multi_nodes_v3.sh
```

Run WeNet recognition directly:

```bash
python wenet/bin/recognize.py \
  --device cuda \
  --modes rnnt_beam_search \
  --config exp/your_model/train.yaml \
  --data_type raw \
  --test_data data/test/data.jsonl \
  --checkpoint exp/your_model/final.pt \
  --beam_size 10 \
  --batch_size 1 \
  --result_dir exp/your_model/decode
```

Decode from a Kaldi-style directory:

```bash
python scripts/decode_from_kaldi_dir.py \
  --data_dir data/test_kaldi \
  --config exp/your_model/train.yaml \
  --checkpoint exp/your_model/final.pt \
  --output_dir exp/your_model/decode_kaldi \
  --modes rnnt_beam_search \
  --decode_nj 8 \
  --device cuda
```

Run the focused unit tests:

```bash
python -m unittest tests.test_decode_from_kaldi_dir
```

## Notes

- Many launch scripts contain local HPC paths and are kept as experiment
  records/templates, not portable one-command demos.
- The `conf`, `data`, `exp`, `icefall`, and `tools` entries in the original
  working directory were local symlinks and are not committed as repository
  content.
- Model weights, checkpoints, shards, audio, and TensorBoard logs are excluded
  from git by design.
