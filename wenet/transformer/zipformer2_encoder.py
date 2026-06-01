import os
import sys
import importlib
from typing import Any, Dict, List, Optional, Tuple

import torch

from wenet.utils.mask import make_pad_mask


class Zipformer2Encoder(torch.nn.Module):
    #[AI修改 开始位置 20260115 适配Zipformer2训练与解码接口] 作用：提供 Zipformer2 最小适配，先跑通训练/解码接口
    def __init__(
        self,
        input_size: int,
        output_size: int = 384,
        encoder_embed_out_channels: Optional[int] = None,
        global_cmvn: Optional[torch.nn.Module] = None,
        zipformer_conf: Optional[Dict[str, Any]] = None,
        subsampling_conf: Optional[Dict[str, Any]] = None,
        icefall_zipformer_dir: Optional[str] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self._output_size = int(output_size)
        self.global_cmvn = global_cmvn

        zipformer_conf = {} if zipformer_conf is None else dict(zipformer_conf)
        subsampling_conf = {} if subsampling_conf is None else dict(
            subsampling_conf)

        #[AI修改 开始位置 20260119 强化icefall模块导入一致性] 作用：避免命名冲突导致误导入非icefall的subsampling/zipformer
        icefall_zipformer_dir = self._ensure_zipformer_importable(icefall_zipformer_dir)
        #[AI修改 开始位置 20260123 彻底规避同名模块导入冲突] 作用：不依赖sys.path查找，直接按目标目录加载并覆盖sys.modules同名缓存
        def _load_py_as_module(module_name: str, base_dir: str) -> Any:
            file_path = os.path.join(base_dir, f"{module_name}.py")
            if not os.path.isfile(file_path):
                raise RuntimeError(f"目录缺少 {module_name}.py: {base_dir}")
            #[AI修改 开始位置 20260123 兼容外部icefall使用软链接组织代码] 作用：不对文件路径做realpath展开，避免软链接指向共享实现导致误判冲突
            file_path = os.path.abspath(file_path)
            #[AI修改 结束位置 20260123 兼容外部icefall使用软链接组织代码]

            sys.modules.pop(module_name, None)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载模块: {module_name} ({file_path})")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded_file = os.path.abspath(getattr(module, "__file__", "") or "")
            if loaded_file and not loaded_file.startswith(os.path.abspath(base_dir) + os.sep):
                raise RuntimeError(f"{module_name} 导入冲突: {loaded_file} (期望前缀 {os.path.abspath(base_dir)})")
            return module

        assert icefall_zipformer_dir is not None
        icefall_zipformer_dir = os.path.realpath(icefall_zipformer_dir)
        _load_py_as_module("encoder_interface", icefall_zipformer_dir)
        _load_py_as_module("scaling", icefall_zipformer_dir)
        subsampling_mod = _load_py_as_module("subsampling", icefall_zipformer_dir)
        zipformer_mod = _load_py_as_module("zipformer", icefall_zipformer_dir)
        #[AI修改 结束位置 20260123 彻底规避同名模块导入冲突]
        #[AI修改 结束位置 20260119 强化icefall模块导入一致性]
        Conv2dSubsampling = getattr(subsampling_mod, "Conv2dSubsampling")
        Zipformer2 = getattr(zipformer_mod, "Zipformer2")

        if encoder_embed_out_channels is None:
            encoder_embed_out_channels = int(self._output_size)

        self.encoder_embed = Conv2dSubsampling(
            in_channels=int(input_size),
            out_channels=int(encoder_embed_out_channels),
            **subsampling_conf,
        )
        self.encoder = Zipformer2(**zipformer_conf)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = decoding_chunk_size
        _ = num_decoding_left_chunks
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        x, x_lens = self.encoder_embed(xs, xs_lens)
        #[AI修改 开始位置 20260119 避免 make_pad_mask() 的 GPU->CPU 同步开销] 作用：传入 max_len，避免 lengths.max().item() 触发同步
        src_key_padding_mask = make_pad_mask(x_lens, x.size(1))  # True 表示 padding
        #[AI修改 结束位置 20260119 避免 make_pad_mask() 的 GPU->CPU 同步开销]
        x = x.permute(1, 0, 2)  # (B,T,D) -> (T,B,D)
        enc_out, enc_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        enc_out = enc_out.permute(1, 0, 2)  # (T,B,D) -> (B,T,D)
        mask = ~make_pad_mask(enc_out_lens, enc_out.size(1)).unsqueeze(1)
        return enc_out, mask

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs_lens = torch.full(
            (xs.size(0),),
            xs.size(1),
            dtype=torch.long,
            device=xs.device,
        )
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        old_chunk_size = getattr(self.encoder, "chunk_size", None)
        old_left_context_frames = getattr(self.encoder, "left_context_frames",
                                          None)
        #[AI修改 开始位置 20260115 模拟流式解码固定chunk与左上下文] 作用：让 simulate_streaming 的 chunk 参数在 Zipformer2 生效，并走 Zipformer2 原生 streaming_forward（逐块前向）
        try:
            if hasattr(self.encoder, "chunk_size"):
                self.encoder.chunk_size = [int(decoding_chunk_size)]
            if hasattr(self.encoder, "left_context_frames"):
                if int(num_decoding_left_chunks) < 0:
                    self.encoder.left_context_frames = [-1]
                elif int(num_decoding_left_chunks) == 0:
                    self.encoder.left_context_frames = [int(decoding_chunk_size)]
                else:
                    self.encoder.left_context_frames = [
                        int(decoding_chunk_size) * int(num_decoding_left_chunks)
                    ]
            if (
                int(num_decoding_left_chunks) >= 0
                and hasattr(self.encoder, "streaming_forward")
                and hasattr(self.encoder, "get_init_states")
            ):
                #[AI修改 开始位置 2026-02-28 修改原因] 修复端到端流式对齐：subsampling必须走streaming_forward+cache，避免时间轴错位导致句错暴涨
                assert int(decoding_chunk_size) > 0
                chunk_size = int(decoding_chunk_size)
                batch_size = int(xs.size(0))

                subsampling_rate = 2
                convnext_left_pad = int(getattr(getattr(self.encoder_embed, "convnext", None), "padding", (3, 3))[0])
                context = 7 + 2 * convnext_left_pad
                stride = subsampling_rate * chunk_size
                decoding_window = stride + context

                if not hasattr(self.encoder_embed, "streaming_forward") or not hasattr(
                    self.encoder_embed, "get_init_states"
                ):
                    raise RuntimeError("Zipformer2严格流式需要encoder_embed支持streaming_forward/get_init_states")

                embed_states = self.encoder_embed.get_init_states(
                    batch_size=batch_size, device=xs.device
                )
                states: List[torch.Tensor] = self.encoder.get_init_states(
                    batch_size=batch_size, device=xs.device
                )

                outputs: List[torch.Tensor] = []
                out_lens = torch.zeros((batch_size,), dtype=torch.long, device=xs.device)
                t_processed_embed = 0
                num_frames = int(xs.size(1))

                for cur in range(0, num_frames - context + 1, stride):
                    end = min(cur + decoding_window, num_frames)
                    real_len = int(end - cur)

                    chunk_xs = xs[:, cur:end, :]
                    if real_len < decoding_window:
                        pad_len = decoding_window - real_len
                        pad = xs.new_zeros((batch_size, pad_len, xs.size(-1)))
                        chunk_xs = torch.cat([chunk_xs, pad], dim=1)
                        chunk_lens = torch.full(
                            (batch_size,), decoding_window, dtype=torch.long, device=xs.device
                        )
                    else:
                        chunk_lens = torch.full(
                            (batch_size,), decoding_window, dtype=torch.long, device=xs.device
                        )

                    x_embed, x_embed_lens, embed_states = self.encoder_embed.streaming_forward(
                        chunk_xs, chunk_lens, embed_states
                    )
                    if int(x_embed.size(1)) != chunk_size:
                        if int(x_embed.size(1)) > chunk_size:
                            x_embed = x_embed[:, :chunk_size, :]
                        else:
                            pad = x_embed.new_zeros((batch_size, chunk_size - int(x_embed.size(1)), x_embed.size(-1)))
                            x_embed = torch.cat([x_embed, pad], dim=1)
                        x_embed_lens = torch.full(
                            (batch_size,), chunk_size, dtype=torch.long, device=xs.device
                        )

                    x_chunk = x_embed.permute(1, 0, 2)  # (B,T,D) -> (T,B,D)
                    x_chunk_lens = x_embed_lens

                    chunk_key_padding_mask = torch.zeros(
                        (batch_size, int(x_chunk.size(0))),
                        dtype=torch.bool,
                        device=xs.device,
                    )

                    left_context_len = int(getattr(self.encoder, "left_context_frames", [0])[0])
                    if left_context_len < 0:
                        left_context_len = 0
                    if left_context_len > 0:
                        hist_valid = min(left_context_len, t_processed_embed)
                        hist_pad = left_context_len - hist_valid
                        if hist_pad > 0:
                            left_pad_mask = torch.ones(
                                (batch_size, hist_pad), dtype=torch.bool, device=xs.device
                            )
                            if hist_valid > 0:
                                left_valid_mask = torch.zeros(
                                    (batch_size, hist_valid), dtype=torch.bool, device=xs.device
                                )
                                left_context_mask = torch.cat([left_pad_mask, left_valid_mask], dim=1)
                            else:
                                left_context_mask = left_pad_mask
                        else:
                            left_context_mask = torch.zeros(
                                (batch_size, left_context_len), dtype=torch.bool, device=xs.device
                            )
                        key_padding_mask = torch.cat([left_context_mask, chunk_key_padding_mask], dim=1)
                    else:
                        key_padding_mask = chunk_key_padding_mask

                    out, out_len, states = self.encoder.streaming_forward(
                        x_chunk,
                        x_chunk_lens,
                        states,
                        key_padding_mask,
                    )
                    outputs.append(out)
                    out_lens = out_lens + out_len
                    t_processed_embed += int(x_chunk.size(0))

                if len(outputs) > 0:
                    enc_out = torch.cat(outputs, dim=0)
                else:
                    enc_out = xs.new_zeros((0, batch_size, self._output_size))

                expected_embed_len = torch.div(xs_lens - 7, 2, rounding_mode="trunc")
                expected_out_len = torch.div(expected_embed_len + 1, 2, rounding_mode="trunc")
                max_expected = int(expected_out_len.max().item()) if expected_out_len.numel() > 0 else 0
                if max_expected > 0 and int(enc_out.size(0)) > max_expected:
                    enc_out = enc_out[:max_expected]
                enc_out_lens = expected_out_len
                #[AI修改 结束位置 2026-02-28 修改原因] 修复端到端流式对齐：subsampling必须走streaming_forward+cache，避免时间轴错位导致句错暴涨
            else:
                x, x_lens = self.encoder_embed(xs, xs_lens)
                #[AI修改 开始位置 20260119 避免 make_pad_mask() 的 GPU->CPU 同步开销] 作用：传入 max_len，避免 lengths.max().item() 触发同步
                src_key_padding_mask = make_pad_mask(x_lens, x.size(1))  # True 表示 padding
                #[AI修改 结束位置 20260119 避免 make_pad_mask() 的 GPU->CPU 同步开销]
                x = x.permute(1, 0, 2)  # (B,T,D) -> (T,B,D)
                enc_out, enc_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        finally:
            if old_chunk_size is not None and hasattr(self.encoder, "chunk_size"):
                self.encoder.chunk_size = old_chunk_size
            if old_left_context_frames is not None and hasattr(
                    self.encoder, "left_context_frames"):
                self.encoder.left_context_frames = old_left_context_frames
        #[AI修改 结束位置 20260115 模拟流式解码固定chunk与左上下文]

        enc_out = enc_out.permute(1, 0, 2)  # (T,B,D) -> (B,T,D)
        mask = ~make_pad_mask(enc_out_lens, enc_out.size(1)).unsqueeze(1)
        return enc_out, mask

    def _ensure_zipformer_importable(self, icefall_zipformer_dir: Optional[str]
                                     ) -> Optional[str]:
        if icefall_zipformer_dir is None:
            icefall_zipformer_dir = os.environ.get(
                "WENET_ICEFALL_ZIPFORMER_DIR",
                os.environ.get("ICEFALL_ZIPFORMER_DIR"),
            )
        if icefall_zipformer_dir is None:
            file_dir = os.path.dirname(__file__)
            candidate_repo_roots = [
                os.path.realpath(os.path.join(file_dir, "..", "..")),
                os.path.realpath(os.path.join(file_dir, "..", "..", "..")),
            ]
            candidates = [
                os.path.join(
                    repo_root,
                    "icefall",
                    "egs",
                    "multi_zh_en",
                    "ASR",
                    "zipformer",
                ) for repo_root in candidate_repo_roots
            ]
            icefall_zipformer_dir = next(
                (p for p in candidates if os.path.isdir(p)),
                candidates[0],
            )
        icefall_zipformer_dir = os.path.realpath(icefall_zipformer_dir)
        if not os.path.isdir(icefall_zipformer_dir):
            raise RuntimeError(
                f"未找到 icefall zipformer 目录: {icefall_zipformer_dir}")
        icefall_repo_root = self._infer_icefall_repo_root(icefall_zipformer_dir)
        if icefall_repo_root is not None and icefall_repo_root not in sys.path:
            sys.path.insert(0, icefall_repo_root)
        if icefall_zipformer_dir not in sys.path:
            sys.path.insert(0, icefall_zipformer_dir)
        if not os.path.isfile(os.path.join(icefall_zipformer_dir,
                                           "zipformer.py")):
            raise RuntimeError(
                f"目录缺少 zipformer.py: {icefall_zipformer_dir}")
        return icefall_zipformer_dir

    def _infer_icefall_repo_root(self, icefall_zipformer_dir: str) -> Optional[str]:
        cur = os.path.realpath(icefall_zipformer_dir)
        for _ in range(10):
            if os.path.isfile(os.path.join(cur, "icefall", "utils.py")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None
    #[AI修改 结束位置 20260115 适配Zipformer2训练与解码接口]
