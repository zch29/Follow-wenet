#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析cuts文件，找出178秒的长音频来源
"""

from lhotse import CutSet
import numpy as np
from collections import defaultdict

# 读取cuts
cuts_path = '/mnt/lustre/hpc_stor01/home/chenghao.zhao/work25/project/data/icefall_data/cuts.train.filtered.jsonl.gz'
print(f"读取cuts文件: {cuts_path}")

cuts = CutSet.from_file(cuts_path)
print(f"总cuts数: {len(cuts)}")

# 统计duration分布
durations = [c.duration for c in cuts]
print(f"\nDuration分布:")
print(f"  最小: {min(durations):.2f}s")
print(f"  最大: {max(durations):.2f}s")
print(f"  平均: {np.mean(durations):.2f}s")
print(f"  中位数: {np.median(durations):.2f}s")
print(f"  95分位: {np.percentile(durations, 95):.2f}s")
print(f"  99分位: {np.percentile(durations, 99):.2f}s")

# 找出超长cut
print(f"\n超过60秒的cuts:")
long_cuts = [c for c in cuts if c.duration > 60]
print(f"数量: {len(long_cuts)} ({len(long_cuts)/len(cuts)*100:.2f}%)")

# 按时长分组
bins = [(0, 30), (30, 60), (60, 120), (120, 180), (180, float('inf'))]
print(f"\n时长区间分布:")
for vmin, vmax in bins:
    count = sum(1 for d in durations if d >= vmin and d < vmax)
    print(f"  [{vmin:3d}s, {vmax:s:5s}): {count:6d} cuts ({count/len(durations)*100:5.1f}%)")

# 分析最长的5个cut
print(f"\n最长的5个cuts:")
sorted_cuts = sorted(cuts, key=lambda c: c.duration, reverse=True)[:5]
for i, cut in enumerate(sorted_cuts, 1):
    print(f"\n  #{i}: duration={cut.duration:.2f}s")
    print(f"      id={cut.id}")
    print(f"      recording_id={cut.recording_id}")
    if hasattr(cut, 'supervisions') and cut.supervisions:
        sup = cut.supervisions[0]
        print(f"      text={sup.text[:100]}...")  # 只显示前100字符

    # 检查是否是拼接的
    if hasattr(cut, 'cuts') and cut.cuts:
        print(f"      >>> 这是一个拼接的cut，由{len(cut.cuts)}个子cuts组成 <<<")
        for j, sub_cut in enumerate(cut.cuts[:3], 1):  # 只显示前3个
            print(f"         子cut#{j}: {sub_cut.duration:.2f}s")
        if len(cut.cuts) > 3:
            print(f"         ... 还有{len(cut.cuts)-3}个子cuts")

# 检查拼接cut的比例
concat_cuts = [c for c in cuts if hasattr(cut, 'cuts') and cut.cuts]
print(f"\n拼接cut统计:")
print(f"  数量: {len(concat_cuts)} ({len(concat_cuts)/len(cuts)*100:.2f}%)")

if concat_cuts:
    concat_durations = [c.duration for c in concat_cuts]
    print(f"  平均时长: {np.mean(concat_durations):.2f}s")
    print(f"  最大时长: {max(concat_durations):.2f}s")
    print(f"  平均子cuts数: {np.mean([len(c.cuts) for c in concat_cuts]):.1f}")

# 找出178秒的cut
target_cut = next((c for c in cuts if abs(c.duration - 178.41) < 0.1), None)
if target_cut:
    print(f"\n找到178.41秒的cut:")
    print(f"  id={target_cut.id}")
    print(f"  recording_id={target_cut.recording_id}")

    if hasattr(target_cut, 'supervisions') and target_cut.supervisions:
        sup = target_cut.supervisions[0]
        print(f"  text={sup.text}")

    if hasattr(target_cut, 'cuts') and target_cut.cuts:
        print(f"  >>> 这是由{len(target_cut.cuts)}个子cuts拼接而成的 <<<")
        for j, sub_cut in enumerate(target_cut.cuts, 1):
            print(f"    子cut#{j:2d}: {sub_cut.duration:7.2f}s - {sub_cut.recording_id}")
            if hasattr(sub_cut, 'supervisions') and sub_cut.supervisions:
                print(f"           text={sub_cut.supervisions[0].text[:50]}...")