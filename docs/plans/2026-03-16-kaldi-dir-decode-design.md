# Kaldi Dir Decode Utility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python entrypoint that accepts a Kaldi-style directory containing `text` and `wav.scp`, converts it to Wenet raw `jsonl`, runs `recognize.py`, and writes decode analysis outputs.

**Architecture:** The new script lives in `scripts/` as an entrypoint layer above the current Wenet decode pipeline. It keeps model decode logic in `wenet/bin/recognize.py`, uses `--data_type raw` plus `override_config dataset_conf.use_precomputed_feat=false`, and adds a thin post-processing stage for merged hypotheses and per-utterance comparison.

**Tech Stack:** Python 3, argparse, subprocess, pathlib, JSONL, pytest, existing `tools/compute-wer-v2.py`

---

### Task 1: Define the utility contract

**Files:**
- Create: `docs/plans/2026-03-16-kaldi-dir-decode-design.md`
- Create: `tests/test_decode_from_kaldi_dir.py`

**Step 1: Write the failing test**

Write a test that creates a temporary Kaldi-style directory with `text` and `wav.scp`, imports helper functions from `scripts/decode_from_kaldi_dir.py`, and asserts that the generated JSONL lines preserve `key`, `wav`, and `txt`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: FAIL because `scripts/decode_from_kaldi_dir.py` does not exist yet.

**Step 3: Write minimal implementation**

Create `scripts/decode_from_kaldi_dir.py` with:
- a parser for Kaldi text files
- a parser for `wav.scp`
- a join function by utterance id
- a JSONL writer

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: PASS

### Task 2: Add decode orchestration

**Files:**
- Create: `scripts/decode_from_kaldi_dir.py`

**Step 1: Write the failing test**

Add a test around argument construction for `recognize.py`, asserting:
- `--data_type raw`
- `--test_data <generated jsonl>`
- `--override_config dataset_conf.use_precomputed_feat=false`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: FAIL because the command builder is missing.

**Step 3: Write minimal implementation**

Add command-building helpers and a `main()` flow that:
- validates inputs
- writes JSONL into an output directory
- shells out to `recognize.py`
- supports decode mode, batch size, beam size, checkpoint, config, and result directory arguments

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: PASS

### Task 3: Add post-decode merge and analysis outputs

**Files:**
- Modify: `scripts/decode_from_kaldi_dir.py`
- Modify: `tests/test_decode_from_kaldi_dir.py`

**Step 1: Write the failing test**

Add a test for post-processing helpers that:
- read a Wenet hypothesis text file
- join it with references from `text`
- emit a tab-separated comparison table with `key`, `ref`, `hyp`, and match status

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: FAIL because the analysis helpers are missing.

**Step 3: Write minimal implementation**

Add helpers to:
- read merged hypothesis text
- create `analysis/compare.tsv`
- create `analysis/summary.txt`
- optionally invoke `tools/compute-wer-v2.py` when available

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: PASS

### Task 4: Verify on real data

**Files:**
- Modify: `scripts/decode_from_kaldi_dir.py` if verification finds issues

**Step 1: Run targeted unit tests**

Run: `pytest tests/test_decode_from_kaldi_dir.py -v`
Expected: PASS

**Step 2: Run a real decode**

Run the new script on `/hpc_stor01/home/chenghao.zhao/work24/zh-multilingual/data/badcase_v2/local_260313/...` with an existing config and checkpoint.

Expected:
- `data.jsonl` created
- Wenet decode output created
- merged `hyp/text`
- `analysis/compare.tsv`
- `analysis/summary.txt`
- `wer` if `compute-wer-v2.py` is available

**Step 3: Fix any issues found in verification**

Keep changes scoped to input handling, command wiring, or output post-processing.
