# -*- coding: utf-8 -*-
import sys
import io
import argparse
import os
# 重定向标准输出到 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="输入文件路径")
    parser.add_argument('n_split', type=int, help="分割数量（≥1）")
    parser.add_argument('output_dir', help="输出目录")
    args = parser.parse_args()

    # 基础校验
    if args.n_split < 1:
        raise ValueError("分割数量必须≥1")
    os.makedirs(args.output_dir, exist_ok=True)

    #[AI修改 开始位置 2026-02-09 修改原因] 将分片逻辑改为真·流式写入，避免大文件 OOM
    output_paths = [
        os.path.join(args.output_dir, f"split_{idx+1}.txt")
        for idx in range(args.n_split)
    ]
    out_files = []
    counts = [0 for _ in range(args.n_split)]
    try:
        for p in output_paths:
            out_files.append(open(p, "w", encoding="utf-8"))

        with open(args.input_file, "r", encoding="utf-8") as in_f:
            written = 0
            for raw_line in in_f:
                line = raw_line.strip()
                if not line:
                    continue
                split_idx = written % args.n_split
                out_files[split_idx].write(line + "\n")
                counts[split_idx] += 1
                written += 1
    finally:
        for f in out_files:
            try:
                f.close()
            except Exception:
                pass

    for idx, p in enumerate(output_paths):
        print(f"生成: {p} ({counts[idx]}行)")
    #[AI修改 结束位置 2026-02-09 修改原因] 将分片逻辑改为真·流式写入，避免大文件 OOM

if __name__ == "__main__":
    main()
