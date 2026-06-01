# Follow-Wenet

Follow-Wenet is an open research fork of WeNet for RNN-T/Transducer and
Zipformer-based automatic speech recognition (ASR) experiments. The project
keeps WeNet's familiar training and decoding workflow, while adding a larger
experimental surface for HAT/RNN-T training, k2-pruned losses, Zipformer2
encoder integration, and All-in-One style multi-mode ASR.

This repository is intended for researchers and engineers who want to study
streaming ASR model design, reproduce RNN-T/HAT experiments, or compare hybrid
CTC/AED/RNN-T decoding strategies in a WeNet-compatible codebase.

## Why This Project

WeNet is a practical open-source ASR toolkit, but its native Transducer path is
not enough for some recent RNN-T/HAT and Zipformer experiments. Follow-Wenet
collects those experimental extensions in one public repository:

- A usable RNN-T/HAT training path inside WeNet.
- A multi-mode joiner inspired by All-in-One ASR, sharing encoder, predictor,
  and joiner behavior across HAT/RNN-T, AED, CTC, and internal-LM modes.
- k2 smoothed/pruned RNN-T loss support to reduce memory pressure for large
  `[B, T, U, V]` HAT tensors.
- A WeNet adapter for icefall Zipformer2, including simulated streaming support.
- Decode utilities for Kaldi-style `text` + `wav.scp` directories.

The project does not claim production-ready benchmarks yet. Its open-source
value is in making the implementation, configuration examples, and workflow
scripts available for inspection, reuse, and further comparison.

## Feature Highlights

- **RNN-T/HAT model path**
  - Extends `wenet/transducer/transducer.py`.
  - Supports standard Transducer training and hybrid loss experiments.
  - Adds stable loss dictionary outputs for easier training diagnostics.

- **Multi-mode All-in-One style joiner**
  - Implemented in `wenet/transducer/joint_optimized.py`.
  - Supports `hat`, `aed`, `ctc`, and `lm` forward modes.
  - Uses gradient checkpointing and pruned HAT forward paths for memory control.

- **k2 integration**
  - Supports smoothed and pruned RNN-T loss when `enable_k2: true`.
  - Avoids always materializing full RNN-T/HAT logits for large experiments.

- **Zipformer2 in WeNet**
  - Adds `wenet/transformer/zipformer2_encoder.py`.
  - Wraps icefall Zipformer2 behind a WeNet encoder interface.
  - Includes stricter module loading to avoid Python import conflicts.
  - Supports simulated streaming with Zipformer states and subsampling caches.

- **Decoding extensions**
  - Adds modes such as `rnnt_beam_search`,
    `rnnt_hat_aed_alsd_beam_search`,
    `rnnt_hat_offline_aed_rescoring`, and
    `rnnt_offline_aed_greedy_search`.
  - Exposes AED score weight, ALSD-style blank gating, AED attention windows,
    HAT label weighting, and repeat suppression controls.

- **Workflow utilities**
  - `scripts/decode_from_kaldi_dir.py` converts a Kaldi-style data directory to
    WeNet raw JSONL, runs decode jobs, merges hypotheses, and writes analysis
    summaries.
  - Focused unit tests are included for the Kaldi directory decode utility.

## Use Cases

- Researching RNN-T/HAT training behavior in a WeNet-compatible project.
- Testing multi-objective ASR training with HAT/RNN-T, AED, CTC, and LM-style
  losses.
- Adapting icefall Zipformer2 ideas into a WeNet pipeline.
- Running controlled decoding experiments with RNN-T, CTC, AED rescoring, and
  streaming-style chunk settings.
- Building small utilities around Kaldi-style ASR evaluation data.

## Screenshots

This is primarily a model-training and decoding repository, so there is no GUI
dashboard in the repo. Suggested screenshot locations for future project pages:

- `docs/assets/training-loss.png`: TensorBoard loss curves.
- `docs/assets/decode-summary.png`: decode comparison or WER summary.
- `docs/assets/model-diagram.png`: encoder/predictor/joiner architecture.

Large TensorBoard event files and experiment images are intentionally not
committed.

## Installation

Follow-Wenet is a research tree, not a polished pip package. Start from a Python
environment that can run WeNet, then add optional dependencies for the
experiments you want.

```bash
git clone https://github.com/zch29/Follow-wenet.git
cd Follow-wenet

# Recommended: use your existing WeNet/PyTorch environment.
python -m pip install torch torchaudio pyyaml
```

Optional dependencies:

- `k2` for `enable_k2: true` and pruned RNN-T loss experiments.
- `icefall` for Zipformer2 experiments.
- Kaldi tools for ark/Kaldi-style feature workflows.
- Internal cluster tooling such as `vc` if you want to reuse the provided HPC
  launch scripts.

For Zipformer2, point the adapter to an icefall Zipformer directory:

```bash
export WENET_ICEFALL_ZIPFORMER_DIR=/path/to/icefall/egs/multi_zh_en/ASR/zipformer
```

## Running

Run the focused unit tests:

```bash
python -m unittest tests.test_decode_from_kaldi_dir
```

Run recognition with an existing model checkpoint:

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

Decode a Kaldi-style directory containing `text` and `wav.scp`:

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

Train with an example configuration after adapting data paths and environment
settings:

```bash
bash run_multi_nodes_v3.sh
```

The launch scripts contain local HPC paths and should be treated as templates.
For portable experiments, start from the configuration snapshots in
`examples/conf/` and adjust data, tokenizer, CMVN, checkpoint, and cluster
settings.

## Technology Stack

- Python
- PyTorch and torchaudio
- WeNet-style model, dataset, and decoding APIs
- k2 for optional pruned RNN-T loss
- icefall Zipformer2 integration
- Kaldi-style data conventions for selected workflows
- Bash scripts for local/HPC experiment orchestration

## Project Structure

```text
.
├── README.md
├── examples/
│   └── conf/                    # Snapshot configs for RNNT, multi-mode, Zipformer
├── scripts/
│   └── decode_from_kaldi_dir.py # Kaldi-style directory decode helper
├── tests/
│   └── test_decode_from_kaldi_dir.py
├── wenet/
│   ├── bin/                     # Training and recognition entrypoints
│   ├── dataset/                 # Dataset and feature processing utilities
│   ├── transducer/              # RNN-T/HAT model, joiner, search code
│   ├── transformer/             # Encoder/decoder modules and Zipformer2 adapter
│   └── utils/                   # Model init, checkpoint, training utilities
├── run_*.sh                     # Training launch templates
├── decode_conformer_ark*.sh     # Decode launch templates
└── path.sh                      # Local environment template
```

Not tracked by design:

- model checkpoints and exported models
- audio, shards, and large feature files
- TensorBoard logs and benchmark outputs
- local symlinked projects such as `data`, `exp`, `conf`, `icefall`, and `tools`

## Roadmap

- Add a minimal public toy recipe that can run without internal data paths.
- Add smaller CI-friendly tests for model initialization and multi-mode joiner
  shape checks.
- Document expected config fields for RNN-T/HAT, k2 pruning, and Zipformer2.
- Add architecture diagrams under `docs/assets/`.
- Publish reproducible experiment notes when public data and compute settings
  are available.
- Separate portable examples from local HPC launch scripts.

## Contributing

Contributions are welcome, especially in areas that make the project easier to
run outside the original research environment.

Recommended contribution workflow:

1. Open an issue describing the bug, experiment, or documentation gap.
2. Keep pull requests focused and small.
3. Include a short explanation of the ASR behavior being changed.
4. Add or update tests when touching utility code or shared model interfaces.
5. Avoid committing checkpoints, audio, TensorBoard logs, private paths, or
   generated experiment outputs.

Before opening a pull request, run:

```bash
python -m unittest tests.test_decode_from_kaldi_dir
git diff --check
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
