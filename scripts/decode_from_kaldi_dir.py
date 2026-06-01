#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


DEFAULT_IMAGE = "docker.v2.aispeech.com/hpc/ai_on_device-hao.li_k2:pytorch2.1.0-cuda11.8-v1.4.0"


def parse_kaldi_text(path):
    records = {}
    with open(str(path), "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError("Invalid Kaldi text line {0}: {1}".format(line_num, line))
            key, text = parts
            records[key] = text
    return records


def parse_wav_scp(path):
    records = {}
    with open(str(path), "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError("Invalid wav.scp line {0}: {1}".format(line_num, line))
            key, wav_path = parts
            records[key] = wav_path
    return records


def build_jsonl_records(data_dir):
    data_dir = Path(data_dir)
    text_map = parse_kaldi_text(data_dir / "text")
    wav_map = parse_wav_scp(data_dir / "wav.scp")

    missing_wav = sorted(set(text_map) - set(wav_map))
    missing_text = sorted(set(wav_map) - set(text_map))
    if missing_wav:
        raise ValueError("Missing wav.scp entries for keys: {0}".format(",".join(missing_wav[:10])))
    if missing_text:
        raise ValueError("Missing text entries for keys: {0}".format(",".join(missing_text[:10])))

    rows = []
    for key in sorted(text_map):
        rows.append({
            "key": key,
            "wav": wav_map[key],
            "txt": text_map[key],
        })
    return rows


def write_jsonl(rows, jsonl_path):
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(jsonl_path), "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_recognize_command(python_bin,
                            recognize_script,
                            config,
                            checkpoint,
                            test_data,
                            result_dir,
                            modes,
                            data_type,
                            batch_size,
                            beam_size,
                            device,
                            num_decoding_left_chunks,
                            decoding_chunk_size,
                            simulate_streaming,
                            ctc_weight,
                            override_config):
    command = [
        python_bin,
        recognize_script,
        "--device", device,
        "--modes",
    ]
    command.extend(modes)
    command.extend([
        "--config", config,
        "--data_type", data_type,
        "--test_data", test_data,
        "--checkpoint", checkpoint,
        "--beam_size", str(beam_size),
        "--batch_size", str(batch_size),
        "--ctc_weight", str(ctc_weight),
        "--result_dir", result_dir,
        "--num_decoding_left_chunks", str(num_decoding_left_chunks),
    ])
    if decoding_chunk_size is not None:
        command.extend(["--decoding_chunk_size", str(decoding_chunk_size)])
    if simulate_streaming:
        command.append("--simulate_streaming")
    for item in override_config:
        command.extend(["--override_config", item])
    return command


def read_hypotheses(path):
    hypotheses = {}
    with open(str(path), "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            key = parts[0]
            hyp = parts[1] if len(parts) > 1 else ""
            hypotheses[key] = hyp
    return hypotheses


def write_analysis(refs, hyps, analysis_dir):
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    compare_path = analysis_dir / "compare.tsv"
    summary_path = analysis_dir / "summary.txt"
    total = 0
    exact_match = 0

    with open(str(compare_path), "w", encoding="utf-8") as fout:
        fout.write("key\tref\thyp\tstatus\n")
        for key in sorted(refs):
            total += 1
            ref = refs.get(key, "")
            hyp = hyps.get(key, "")
            status = "OK" if ref == hyp else "ERR"
            if status == "OK":
                exact_match += 1
            fout.write("{0}\t{1}\t{2}\t{3}\n".format(key, ref, hyp, status))

    exact_rate = 0.0
    if total > 0:
        exact_rate = float(exact_match) / float(total)
    with open(str(summary_path), "w", encoding="utf-8") as fout:
        fout.write("total\t{0}\n".format(total))
        fout.write("exact_match\t{0}\n".format(exact_match))
        fout.write("exact_match_rate\t{0:.6f}\n".format(exact_rate))
    return compare_path, summary_path


def run_command(command, cwd=None):
    print("Running:", " ".join(shlex.quote(item) for item in command))
    subprocess.run(command, cwd=cwd, check=True)


def split_jsonl(split_script, jsonl_path, decode_nj, split_dir, python_bin):
    split_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        split_script,
        str(jsonl_path),
        str(decode_nj),
        str(split_dir),
    ]
    run_command(command)


def merge_hypotheses(log_dir, mode, decode_nj, merged_text_path):
    merged_text_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(merged_text_path), "w", encoding="utf-8") as fout:
        for idx in range(1, decode_nj + 1):
            part_path = log_dir / str(idx) / mode / "text"
            if not part_path.exists():
                raise FileNotFoundError("Missing hypothesis shard: {0}".format(part_path))
            with open(str(part_path), "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)


def maybe_run_wer(tool_path, ref_text, hyp_text, wer_path, python_bin):
    tool_path = Path(tool_path)
    if not tool_path.exists():
        return False
    command = [
        python_bin,
        str(tool_path),
        "--char=1",
        "--v=1",
        str(ref_text),
        str(hyp_text),
    ]
    with open(str(wer_path), "w", encoding="utf-8") as fout:
        subprocess.run(command, stdout=fout, stderr=subprocess.STDOUT, check=True)
    return True


def build_vc_submit_command(cpu_cmd_prefix, decode_nj, log_dir, recognize_command):
    recognize_str = " ".join(shlex.quote(item) for item in recognize_command)
    return (
        "{cpu_cmd} -j wenet-decode -pj none JOB=1:{jobs} {log_dir}/decode.JOB.log --cmd "
        "\"{recognize}\" 2>&1 | tee {log_dir}/log.JOB"
    ).format(
        cpu_cmd=cpu_cmd_prefix,
        jobs=decode_nj,
        log_dir=shlex.quote(str(log_dir)),
        recognize=recognize_str.replace("split_1.txt", "split_JOB.txt").replace("/1/", "/JOB/"),
    )


def build_vc_submit_command_from_args(args, decode_nj, log_dir, recognize_command):
    base = [
        "vc", "submit",
        "-p", args.vc_partition,
        "-i", args.image,
        "-c", str(args.vc_cpu_per_task),
        "-m", args.vc_mem_per_task,
    ]
    if args.vc_gpu_per_task is not None:
        base.extend(["-g", str(args.vc_gpu_per_task)])
    base.extend([
        "JOB=1:{0}".format(decode_nj),
        str(log_dir / "decode.JOB.log"),
    ])
    if args.vc_project:
        base.extend(["-pj", args.vc_project])
    if args.vc_job_name:
        base.extend(["-j", args.vc_job_name])
    base.extend([
        "--cmd",
        " ".join(shlex.quote(item) for item in recognize_command),
    ])
    return base


def parse_args():
    parser = argparse.ArgumentParser(description="Decode a Kaldi-style data dir with Wenet raw-audio input.")
    parser.add_argument("--data_dir", required=True, help="Directory containing text and wav.scp")
    parser.add_argument("--config", required=True, help="Wenet config yaml")
    parser.add_argument("--checkpoint", required=True, help="Wenet checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--modes", nargs="+", default=["rnnt_beam_search"], help="Decode modes")
    parser.add_argument("--decode_nj", type=int, default=1, help="Number of split jobs")
    parser.add_argument("--beam_size", type=int, default=10, help="Beam size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--ctc_weight", type=float, default=0.1, help="CTC weight")
    parser.add_argument("--device", default="cpu", help="Decode device")
    parser.add_argument("--num_decoding_left_chunks", type=int, default=5, help="Left chunk count")
    parser.add_argument("--decoding_chunk_size", type=int, default=16, help="Chunk size")
    parser.add_argument("--simulate_streaming", action="store_true", help="Enable simulate streaming")
    parser.add_argument("--python_bin", default="python", help="Python executable used inside decode environment")
    parser.add_argument("--host_python_bin", default=sys.executable or "python3", help="Python executable used on the host for helper scripts")
    parser.add_argument("--split_script", default="split_lines.py", help="Path to split_lines.py")
    parser.add_argument("--recognize_script", default="wenet/bin/recognize.py", help="Path to recognize.py")
    parser.add_argument("--wer_tool", default="tools/compute-wer-v2.py", help="Path to compute-wer-v2.py")
    parser.add_argument("--submit", choices=["local", "vc"], default="local", help="How to run decode")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for vc submit")
    parser.add_argument("--cpu_cmd", default="", help="Optional full vc submit CPU prefix")
    parser.add_argument("--vc_partition", default="pdcpu", help="Partition for vc submit")
    parser.add_argument("--vc_project", default="", help="Project for vc submit, for example a-i-o")
    parser.add_argument("--vc_job_name", default="wenet-decode", help="Job name for vc submit")
    parser.add_argument("--vc_cpu_per_task", type=int, default=2, help="CPU per task for vc submit")
    parser.add_argument("--vc_mem_per_task", default="24G", help="Memory per task for vc submit")
    parser.add_argument("--vc_gpu_per_task", type=int, default=None, help="GPU per task for vc submit")
    parser.add_argument("--work_dir", default=".", help="Working directory for decode commands")
    parser.add_argument(
        "--override_config",
        action="append",
        default=[],
        help="Extra override_config values for recognize.py",
    )
    return parser.parse_args()


def ensure_required_inputs(data_dir):
    data_dir = Path(data_dir)
    missing = []
    for name in ["text", "wav.scp"]:
        if not (data_dir / name).exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError("Missing required files in {0}: {1}".format(data_dir, ",".join(missing)))


def run_local_decode(args, split_dir, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, args.decode_nj + 1):
        split_path = split_dir / "split_{0}.txt".format(idx)
        shard_result_dir = log_dir / str(idx)
        command = build_recognize_command(
            python_bin=args.python_bin,
            recognize_script=args.recognize_script,
            config=args.config,
            checkpoint=args.checkpoint,
            test_data=str(split_path),
            result_dir=str(shard_result_dir),
            modes=args.modes,
            data_type="raw",
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            device=args.device,
            num_decoding_left_chunks=args.num_decoding_left_chunks,
            decoding_chunk_size=args.decoding_chunk_size,
            simulate_streaming=args.simulate_streaming,
            ctc_weight=args.ctc_weight,
            override_config=args.override_config + ["dataset_conf.use_precomputed_feat=false"],
        )
        run_command(command, cwd=args.work_dir)


def run_vc_decode(args, split_dir, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split_1.txt"
    shard_result_dir = log_dir / "1"
    recognize_command = build_recognize_command(
        python_bin=args.python_bin,
        recognize_script=args.recognize_script,
        config=args.config,
        checkpoint=args.checkpoint,
        test_data=str(split_path),
        result_dir=str(shard_result_dir),
        modes=args.modes,
        data_type="raw",
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        device=args.device,
        num_decoding_left_chunks=args.num_decoding_left_chunks,
        decoding_chunk_size=args.decoding_chunk_size,
        simulate_streaming=args.simulate_streaming,
        ctc_weight=args.ctc_weight,
        override_config=args.override_config + ["dataset_conf.use_precomputed_feat=false"],
    )
    if args.cpu_cmd:
        submit_command = build_vc_submit_command(args.cpu_cmd, args.decode_nj, log_dir, recognize_command)
        run_command(["bash", "-lc", submit_command], cwd=args.work_dir)
        return
    submit_command = build_vc_submit_command_from_args(
        args=args,
        decode_nj=args.decode_nj,
        log_dir=log_dir,
        recognize_command=recognize_command,
    )
    submit_command_str = " ".join(shlex.quote(item) for item in submit_command)
    run_command(["bash", "-lc", submit_command_str], cwd=args.work_dir)


def main():
    args = parse_args()
    ensure_required_inputs(args.data_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "data.jsonl"
    split_dir = output_dir / "split"
    log_dir = output_dir / "log"

    refs = parse_kaldi_text(Path(args.data_dir) / "text")
    rows = build_jsonl_records(args.data_dir)
    write_jsonl(rows, jsonl_path)
    split_jsonl(args.split_script, jsonl_path, args.decode_nj, split_dir, args.host_python_bin)

    if args.submit == "vc":
        run_vc_decode(args, split_dir, log_dir)
    else:
        run_local_decode(args, split_dir, log_dir)

    for mode in args.modes:
        mode_dir = output_dir / mode
        merged_text = mode_dir / "text"
        merge_hypotheses(log_dir, mode, args.decode_nj, merged_text)
        hyps = read_hypotheses(merged_text)
        analysis_dir = mode_dir / "analysis"
        write_analysis(refs, hyps, analysis_dir)
        maybe_run_wer(args.wer_tool, Path(args.data_dir) / "text", merged_text, mode_dir / "wer", args.host_python_bin)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("ERROR:", ex, file=sys.stderr)
        raise
