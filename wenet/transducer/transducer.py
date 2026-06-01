from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from wenet.transducer.predictor import PredictorBase
from wenet.transducer.search.greedy_search import basic_greedy_search
from wenet.transducer.search.prefix_beam_search import PrefixBeamSearch
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_blank, add_sos_eos,
                                reverse_pad_list, TORCH_NPU_AVAILABLE)


class Transducer(ASRModel):
    """Transducer-ctc-attention hybrid Encoder-Predictor-Decoder model"""

    def __init__(
        self,
        rnnt: False,
        vocab_size: int,
        blank: int,
        encoder: nn.Module,
        predictor: PredictorBase,
        joint: nn.Module,
        attention_decoder: Optional[Union[TransformerDecoder,
                                          BiTransformerDecoder]] = None,
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        transducer_weight: float = 1.0,
        attention_weight: float = 0.0,
        enable_k2: bool = False,
        delay_penalty: float = 0.0,
        warmup_steps: float = 25000,
        lm_only_scale: float = 0.25,
        am_only_scale: float = 0.0,
        lm_loss_weight: float = 0.1,
        special_tokens: dict = None,
    ) -> None:
        assert attention_weight + ctc_weight + transducer_weight == 1.0
        super().__init__(vocab_size,
                         encoder,
                         attention_decoder,
                         ctc,
                         ctc_weight,
                         ignore_id,
                         reverse_weight,
                         lsm_weight,
                         length_normalized_loss,
                         special_tokens=special_tokens)

        self.blank = blank
        self.transducer_weight = transducer_weight
        self.attention_decoder_weight = 1 - self.transducer_weight - self.ctc_weight
        # 论文中 LM mode 的 loss 单独用系数 λ 加权（默认 0.1）
        self.lm_loss_weight = lm_loss_weight

        self.predictor = predictor
        self.joint = joint
        self.bs = None

        # k2 rnnt loss
        #[AI修改 开始位置 20260112 添加HAT对齐mask到AED训练]
        # 作用：保留 HAT 对齐信息，用于训练阶段给 AED 构造对齐感知的 attention mask
        # 默认不开启，避免对现有实验产生任何影响；需要时可在 config 中显式打开。
        self.use_hat_alignment_for_aed: bool = False
        # 对齐窗口宽度（以帧为单位）：允许 AED 在 t_u±hat_align_window 内做 cross-attention
        self.hat_align_window: int = 4
        #[AI修改 结束位置 20260112 添加HAT对齐mask到AED训练]

        # k2 rnnt loss
        self.enable_k2 = enable_k2
        self.delay_penalty = delay_penalty
        if delay_penalty != 0.0:
            assert self.enable_k2 is True
        self.lm_only_scale = lm_only_scale
        self.am_only_scale = am_only_scale
        self.warmup_steps = warmup_steps
        self.simple_am_proj: Optional[nn.Linear] = None
        self.simple_lm_proj: Optional[nn.Linear] = None
        if self.enable_k2:
            self.simple_am_proj = torch.nn.Linear(self.encoder.output_size(),
                                                  vocab_size)
            self.simple_lm_proj = torch.nn.Linear(self.predictor.output_size(),
                                                  vocab_size)

        # Note(Mddct): decoder also means predictor in transducer,
        # but here decoder is attention decoder
        del self.criterion_att
        if attention_decoder is not None:
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + predictor + joint + loss
        """
        self.device = device
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        steps = batch.get('steps', 0)
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)

        # Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        if getattr(self.joint, "is_multi_mode_joiner", False):
            # 多模式联合器训练路径：
            # - 不再依赖独立的 attention decoder 分支（避免多 decoder 反复计算状态）
            # - 用同一个 joiner 在不同 mode 下产生输出，并分别计算对应的 loss
            # - 这里实现的训练目标与论文 “All-in-One ASR” 保持一致：
            #   L = L_HAT + L_AED + L_CTC + λ * L_LM（权重在 config 中控制）
            # predictor_out: [B, U+1, D_pred]
            # add_blank 会在 token 序列左侧预置一个 blank，使 RNNT/HAT 的 U 维对齐
            ys_in_pad = add_blank(text, self.blank, self.ignore_id)
            predictor_out = self.predictor(ys_in_pad)

            # torchaudio.rnnt_loss 要求 symbols 为 int32，且 padding 位置不能是 ignore_id
            rnnt_text = text.to(torch.int64)
            rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                    rnnt_text).to(torch.int32)
            rnnt_text_lengths = text_lengths.to(torch.int32)
            encoder_out_lens_i32 = encoder_out_lens.to(torch.int32)

            # 1) HAT/Transducer 模式（论文 II-A，核心显存瓶颈在 [B,T,U,V]）
            #[AI修改 开始位置 20260115 启用k2_pruning避免大张量] 作用：在 multi_mode_joiner 下启用 k2 rnnt pruning，避免构造完整 [B,T,U,V]
            if self.enable_k2:
                try:
                    import k2
                except ImportError as e:
                    raise RuntimeError(
                        "enable_k2=True 但未找到 k2，请检查 docker 环境") from e

                if self.simple_lm_proj is None or self.simple_am_proj is None:
                    raise RuntimeError(
                        "enable_k2=True 但 simple_*_proj 未初始化，检查模型构造逻辑")

                delay_penalty = self.delay_penalty
                if steps < 2 * self.warmup_steps:
                    delay_penalty = 0.00

                boundary = torch.zeros((encoder_out.size(0), 4),
                                       dtype=torch.int64,
                                       device=encoder_out.device)
                boundary[:, 3] = encoder_out_lens
                boundary[:, 2] = text_lengths

                rnnt_text_i64 = torch.where(text == self.ignore_id, 0, text)
                lm = self.simple_lm_proj(predictor_out)
                am = self.simple_am_proj(encoder_out)

                amp_autocast = torch.cuda.amp.autocast
                if "npu" in self.device.__str__() and TORCH_NPU_AVAILABLE:
                    amp_autocast = torch.npu.amp.autocast

                with amp_autocast(enabled=False):
                    simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                        lm=lm.float(),
                        am=am.float(),
                        symbols=rnnt_text_i64,
                        termination_symbol=self.blank,
                        lm_only_scale=self.lm_only_scale,
                        am_only_scale=self.am_only_scale,
                        boundary=boundary,
                        reduction="sum",
                        return_grad=True,
                        delay_penalty=delay_penalty,
                    )

                ranges = k2.get_rnnt_prune_ranges(
                    px_grad=px_grad,
                    py_grad=py_grad,
                    boundary=boundary,
                    s_range=5,
                )

                # 注意：MultiModeJoiner.forward 已增加对 pruned 输入([B,T,S,D]) 的支持
                am_pruned, lm_pruned = k2.do_rnnt_pruning(
                    am=self.joint.enc_ffn(encoder_out),
                    lm=self.joint.pred_ffn(predictor_out),
                    ranges=ranges,
                )
                logits_pruned = self.joint(am_pruned,
                                           lm_pruned,
                                           pre_project=False,
                                           mode="hat")

                with amp_autocast(enabled=False):
                    pruned_loss = k2.rnnt_loss_pruned(
                        logits=logits_pruned.float(),
                        symbols=rnnt_text_i64,
                        ranges=ranges,
                        termination_symbol=self.blank,
                        boundary=boundary,
                        reduction="sum",
                        delay_penalty=delay_penalty,
                    )

                simple_loss_scale = 0.5
                if steps < self.warmup_steps:
                    simple_loss_scale = (1.0 - (steps / self.warmup_steps) *
                                         (1.0 - simple_loss_scale))
                pruned_loss_scale = 1.0
                if steps < self.warmup_steps:
                    pruned_loss_scale = 0.1 + 0.9 * (steps / self.warmup_steps)

                loss_hat = (simple_loss_scale * simple_loss +
                            pruned_loss_scale * pruned_loss)
                loss_hat = loss_hat / encoder_out.size(0)
            else:
                hat_logp = self.joint(encoder_out,
                                      predictor_out,
                                      pre_project=True,
                                      mode="hat")
                loss_hat = torchaudio.functional.rnnt_loss(
                    hat_logp,
                    rnnt_text,
                    encoder_out_lens_i32,
                    rnnt_text_lengths,
                    blank=self.blank,
                    reduction="mean",
                )
            #[AI修改 结束位置 20260115 启用k2_pruning避免大张量]

            #[AI修改 开始位置 20260112 添加HAT对齐mask到AED训练]
            # 2') 基于 HAT 对齐路径为 AED 构造 attention mask（训练/推理对齐约束一致）
            # 说明：
            # - 默认 self.use_hat_alignment_for_aed=False，不改变现有训练行为；
            # - 打开开关后，会额外运行一次基于 HAT logp 的 Viterbi 对齐，用于产生 t_u；
            # - 然后根据 t_u 和 hat_align_window 生成 attn_mask[B,U,T]，仅允许在对齐附近做注意力。
            attn_mask: Optional[torch.Tensor] = None
            if getattr(self, "use_hat_alignment_for_aed", False):
                # 训练阶段对齐只用于约束 AED，不参与反向传播，放在 no_grad 区域
                with torch.no_grad():
                    # 非 k2 路径下已经有完整的 hat_logp，可直接复用以避免重复计算
                    if "hat_logp" in locals():
                        hat_logp_for_align = hat_logp
                    else:
                        # k2 路径下为了省显存没有构造 [B,T,U,V]，此处显式计算一次用于对齐
                        # 注意：这会增加显存/算力开销，仅在明确需要严格对齐时打开开关
                        hat_logp_for_align = self.joint(encoder_out,
                                                        predictor_out,
                                                        pre_project=True,
                                                        mode="hat")

                    attn_mask = self._build_aed_attn_mask_from_hat(
                        hat_logp_for_align,
                        rnnt_text,
                        encoder_out_lens_i32,
                        rnnt_text_lengths,
                        self.hat_align_window,
                    )
            #[AI修改 结束位置 20260112 添加HAT对齐mask到AED训练]

            # 2) AED 模式：预测第 u 个 token 时使用 predictor_out[:, u, :]
            # 这里丢弃 predictor_out 的第 0 个位置（对应预置 blank）
            pred_u = predictor_out[:, 1:, :]
            aed_logits = self.joint(encoder_out,
                                    pred_u,
                                    pre_project=True,
                                    mode="aed",
                                    attn_mask=attn_mask)
            # CE：按 token 数归一化，避免不同 batch 的 padding 量影响 loss 尺度
            loss_aed = F.cross_entropy(aed_logits.reshape(-1, self.vocab_size),
                                       text.reshape(-1),
                                       ignore_index=self.ignore_id,
                                       reduction="sum")
            denom = torch.clamp(text_lengths.sum(), min=1).to(loss_aed.dtype)
            loss_aed = loss_aed / denom

            # 3) CTC 模式：CTC 目标不能含 ignore_id，这里把 padding 替换为 blank
            ctc_targets = torch.where(text == self.ignore_id,
                                      torch.tensor(self.blank,
                                                   device=text.device,
                                                   dtype=text.dtype), text)
            loss_ctc_mode = self.joint.ctc_loss_fn(encoder_out,
                                                   encoder_out_lens,
                                                   ctc_targets,
                                                   text_lengths)

            # 4) LM 模式：不看 encoder，只用 predictor 预测下一个 token（内部 LM）
            lm_logits = self.joint(None, pred_u, pre_project=True, mode="lm")
            loss_lm = F.cross_entropy(lm_logits.reshape(-1, self.vocab_size),
                                      text.reshape(-1),
                                      ignore_index=self.ignore_id,
                                      reduction="sum")
            loss_lm = loss_lm / denom

            # 总 loss：三种 ASR 模式用配置的权重，LM 用单独的 λ
            loss = (self.transducer_weight * loss_hat +
                    self.attention_decoder_weight * loss_aed +
                    self.ctc_weight * loss_ctc_mode +
                    self.lm_loss_weight * loss_lm)
            return {
                'loss': loss,
                'loss_rnnt': loss_hat,
                'loss_aed': loss_aed,
                'loss_ctc_mode': loss_ctc_mode,
                'loss_lm': loss_lm,
                'th_accuracy': None,
            }

        #[AI修改 开始位置 20260206 修复transducer_joint训练forward无返回] 作用：非multi_mode_joiner时返回标准loss_dict，避免loss_dict为None

        # compute_loss
        loss_rnnt = self._compute_loss(encoder_out,
                                       encoder_out_lens,
                                       encoder_mask,
                                       text,
                                       text_lengths,
                                       steps=steps)

        loss = self.transducer_weight * loss_rnnt
        # optional attention decoder
        loss_att: Optional[torch.Tensor] = None
        if self.attention_decoder_weight != 0.0 and self.decoder is not None:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            acc_att = None

        # optional ctc
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(encoder_out, encoder_out_lens, text,
                                   text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc.sum()
        if loss_att is not None:
            loss = loss + self.attention_decoder_weight * loss_att.sum()
        # NOTE: 'loss' must be in dict
        return {
            'loss': loss,
            'loss_att': loss_att,
            'loss_ctc': loss_ctc,
            'loss_rnnt': loss_rnnt,
            'th_accuracy': acc_att,
        }

        #[AI修改 结束位置 20260206 修复transducer_joint训练forward无返回]

    #[AI修改 开始位置 20260112 添加HAT对齐mask到AED训练]
    def _build_aed_attn_mask_from_hat(
        self,
        hat_logp: torch.Tensor,
        rnnt_text: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text_lengths: torch.Tensor,
        hat_align_window: int,
    ) -> torch.Tensor:
        """
        基于 HAT/RNNT logp Viterbi 对齐结果，为 AED 构造对齐感知的 attention mask。

        输入：
        - hat_logp: [B, T, U+1, V]，HAT/RNNT 的 log 概率（包含 blank）
        - rnnt_text: [B, U_max]，int32，padding 位置已替换为 0
        - encoder_out_lens: [B]，每个样本 encoder 有效帧长
        - text_lengths: [B]，每个样本 token 数 U
        - hat_align_window: 对齐窗口半径 W，允许 t ∈ [t_u-W, t_u+W]

        输出：
        - attn_mask: [B, U_max, T_max]，bool，True 表示允许 AED attend
        """
        # 为了实现简洁、稳定，这里在 CPU 上做逐样本 Viterbi，对梯度无影响
        B, T_max, U_plus1, _ = hat_logp.shape
        U_max = rnnt_text.size(1)
        device = hat_logp.device

        attn_mask = torch.zeros((B, U_max, T_max),
                                dtype=torch.bool,
                                device=device)
        # 防止无意义的大窗口导致越界，这里做一个下限约束
        window = max(int(hat_align_window), 0)

        hat_logp_cpu = hat_logp.detach().cpu()
        rnnt_text_cpu = rnnt_text.detach().cpu()
        enc_lens = encoder_out_lens.to(torch.int64).tolist()
        txt_lens = text_lengths.to(torch.int64).tolist()

        import math
        import numpy as np

        for b in range(B):
            T_b = int(enc_lens[b])
            U_b = int(txt_lens[b])
            if T_b <= 0 or U_b <= 0:
                continue

            # 当前样本的有效区域
            T_b = min(T_b, T_max)
            U_b = min(U_b, U_plus1 - 1, U_max)

            # 取出当前样本的 logp 与标签（转为 numpy，便于 Viterbi 动态规划）
            hat_b = hat_logp_cpu[b, :T_b, :U_b + 1, :].numpy()  # [T_b, U_b+1, V]
            y_b = rnnt_text_cpu[b, :U_b].numpy()  # [U_b]

            # DP 状态：alpha[t,u] 表示到达 (t,u) 的最佳 log 概率
            # 状态网格大小为 (T_b+1, U_b+1)，其中 t∈[0,T_b], u∈[0,U_b]
            neg_inf = -1e9
            alpha = np.full((T_b + 1, U_b + 1), neg_inf, dtype=np.float32)
            prev_t = np.zeros((T_b + 1, U_b + 1), dtype=np.int32)
            prev_u = np.zeros((T_b + 1, U_b + 1), dtype=np.int32)
            prev_type = np.zeros((T_b + 1, U_b + 1), dtype=np.int8)  # 0: blank, 1: label

            alpha[0, 0] = 0.0

            blank_id = int(self.blank)

            # 前向 DP：按照标准 RNNT/HAT 拓扑计算最优路径
            for t in range(T_b):
                for u in range(U_b + 1):
                    cur_val = alpha[t, u]
                    if cur_val <= neg_inf / 2:
                        continue

                    # blank 转移：(t,u) -> (t+1,u)，在 t 帧发 blank
                    blank_logp = float(hat_b[t, u, blank_id])
                    cand_blank = cur_val + blank_logp
                    if cand_blank > alpha[t + 1, u]:
                        alpha[t + 1, u] = cand_blank
                        prev_t[t + 1, u] = t
                        prev_u[t + 1, u] = u
                        prev_type[t + 1, u] = 0

                    # label 转移：(t,u) -> (t,u+1)，在 t 帧发出标签 y_u
                    if u < U_b:
                        label_id = int(y_b[u])
                        label_logp = float(hat_b[t, u, label_id])
                        cand_label = cur_val + label_logp
                        if cand_label > alpha[t, u + 1]:
                            alpha[t, u + 1] = cand_label
                            prev_t[t, u + 1] = t
                            prev_u[t, u + 1] = u
                            prev_type[t, u + 1] = 1

            # 回溯得到每个 token 的对齐时间 t_u
            t = T_b
            u = U_b
            t_u = np.zeros((U_b, ), dtype=np.int32)
            # 为避免极端情况下死循环，这里加一个最大步数保护
            max_steps = (T_b + U_b + 2) * 2
            steps = 0
            while not (t == 0 and u == 0) and steps < max_steps:
                pt = prev_t[t, u]
                pu = prev_u[t, u]
                ptype = prev_type[t, u]
                if ptype == 1 and pu < U_b:
                    # label 步：在时间 pt 发出第 pu 个 token
                    t_u[pu] = pt
                t, u = pt, pu
                steps += 1

            # 根据 t_u 与窗口生成当前样本的 attn_mask
            for u_idx in range(U_b):
                center = int(t_u[u_idx])
                left = max(center - window, 0)
                right = min(center + window, T_b - 1)
                if right >= left:
                    attn_mask[b, u_idx, left:right + 1] = True

        return attn_mask
    #[AI修改 结束位置 20260112 添加HAT对齐mask到AED训练]

    def init_bs(self):
        if self.bs is None:
            self.bs = PrefixBeamSearch(self.encoder, self.predictor,
                                       self.joint, self.ctc, self.blank)

    #[AI修改 开始位置 20260112 新增HAT流式离线AED重打分解码]
    def beam_search_nbest(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7,
        nbest: int = 0,
    ) -> Tuple[List[List[int]], List[float], torch.Tensor]:
        """
        返回 nbest 的 HAT/RNNT beam search 结果，并返回 encoder_out 供离线 AED 复用。

        说明：
        - 不改动现有 beam_search 的返回形式，避免影响现有解码功能；
        - 该接口仅供“离线 AED 重打分/重排序”模式使用。
        """
        self.init_bs()
        beams, encoder_out = self.bs.prefix_beam_search(
            speech,
            speech_lengths,
            decoding_chunk_size,
            beam_size,
            num_decoding_left_chunks,
            simulate_streaming,
            ctc_weight,
            transducer_weight,
        )
        k = int(nbest) if int(nbest) > 0 else int(beam_size)
        k = max(min(k, len(beams)), 1)
        hyps: List[List[int]] = []
        scores: List[float] = []
        for s in beams[:k]:
            hyp = s.hyp[1:]
            hyps.append([int(x) for x in hyp])
            scores.append(float(s.score))
        return hyps, scores, encoder_out

    def offline_aed_logprob(
        self,
        encoder_out: torch.Tensor,
        tokens: List[int],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        计算给定 tokens 的离线 AED log 概率和（teacher-forcing）。

        输入：
        - encoder_out: [1, T, D]
        - tokens: 预测序列（不含起始 blank）
        - attn_mask: [1, U, T]，可选，对齐窗口约束（不传则为全局离线 attention）
        """
        if len(tokens) == 0:
            return 0.0
        device = encoder_out.device
        ys = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1,U]
        ys_in_pad = torch.full((1, ys.size(1) + 1),
                               int(self.blank),
                               device=device,
                               dtype=torch.long)
        ys_in_pad[:, 1:] = ys
        predictor_out = self.predictor(ys_in_pad)  # [1,U+1,D]
        pred_u = predictor_out[:, 1:, :]  # [1,U,D]
        aed_logits = self.joint(encoder_out,
                                pred_u,
                                pre_project=True,
                                mode="aed",
                                attn_mask=attn_mask)  # [1,U,V]
        aed_logp = aed_logits.log_softmax(dim=-1)  # [1,U,V]
        gather_index = ys.unsqueeze(-1)  # [1,U,1]
        token_logp = aed_logp.gather(dim=-1, index=gather_index).squeeze(-1)  # [1,U]
        return float(token_logp.sum().item())

    def offline_aed_greedy_search_from_encoder(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 0,
        eos_id: int = -1,
        aed_attend_window: int = 0,
        max_consecutive_repeats: int = 0,
    ) -> List[int]:
        if encoder_out.size(0) != 1:
            raise ValueError("offline_aed_greedy_search 仅支持 batch=1")
        T = int(encoder_out.size(1))
        if max_len <= 0:
            max_len = max(1, 2 * T)
        max_len = int(max_len)
        if max_len < 1:
            raise ValueError("max_len 需 >= 1")
        if max_consecutive_repeats < 0:
            raise ValueError("max_consecutive_repeats 需 >= 0")

        device = encoder_out.device
        cache = self.predictor.init_state(1, method="zero", device=device)
        pre_t = torch.tensor([int(self.blank)], device=device, dtype=torch.int)
        tokens: List[int] = []

        for _ in range(max_len):
            padding = torch.zeros(1, 1, device=device)
            pred_out, cache = self.predictor.forward_step(pre_t.unsqueeze(-1),
                                                          padding, cache)
            attn_mask = None
            if aed_attend_window > 0:
                w = int(aed_attend_window)
                if T > w:
                    attn_mask = torch.zeros((1, 1, T),
                                            device=device,
                                            dtype=torch.bool)
                    attn_mask[:, :, T - w:] = True
            aed_logits = self.joint(encoder_out,
                                    pred_out,
                                    pre_project=True,
                                    mode="aed",
                                    attn_mask=attn_mask)  # [1,1,V]
            logp = aed_logits.squeeze(1).log_softmax(dim=-1)  # [1,V]
            if max_consecutive_repeats > 0 and len(tokens) > 0:
                last_tok = int(tokens[-1])
                run_len = 1
                for j in range(len(tokens) - 2, -1, -1):
                    if int(tokens[j]) == last_tok:
                        run_len += 1
                    else:
                        break
                if run_len >= int(max_consecutive_repeats):
                    logp[0, last_tok] = -1e9
            next_tok = int(torch.argmax(logp, dim=-1).item())
            if eos_id >= 0 and next_tok == int(eos_id):
                break
            tokens.append(next_tok)
            pre_t = torch.tensor([next_tok], device=device, dtype=torch.int)
        return tokens
    #[AI修改 结束位置 20260112 新增HAT流式离线AED重打分解码]

    def _cal_transducer_score(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        hyps_lens: torch.Tensor,
        hyps_pad: torch.Tensor,
    ):
        # ignore id -> blank, add blank at head
        hyps_pad_blank = add_blank(hyps_pad, self.blank, self.ignore_id)
        xs_in_lens = encoder_mask.squeeze(1).sum(1).int()

        # 1. Forward predictor
        predictor_out = self.predictor(hyps_pad_blank)
        # 2. Forward joint
        if getattr(self.joint, "is_multi_mode_joiner", False):
            # 多模式联合器下，用 HAT 模式的 joiner 计算 RNNT/HAT 分数
            joint_out = self.joint(encoder_out,
                                   predictor_out,
                                   pre_project=True,
                                   mode="hat")
        else:
            joint_out = self.joint(encoder_out, predictor_out)
        rnnt_text = hyps_pad.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                rnnt_text).to(torch.int32)
        # 3. Compute transducer loss
        loss_td = torchaudio.functional.rnnt_loss(joint_out,
                                                  rnnt_text,
                                                  xs_in_lens,
                                                  hyps_lens.int(),
                                                  blank=self.blank,
                                                  reduction='none')
        return loss_td * -1

    def _cal_attn_score(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        hyps_pad: torch.Tensor,
        hyps_lens: torch.Tensor,
    ):
        # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad

        # td_score = loss_td * -1
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            self.reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        return decoder_out, r_decoder_out

    def beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7,
    ):
        """beam search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            ctc_weight (float): ctc probability weight in transducer
                prefix beam search.
                final_prob = ctc_weight * ctc_prob + transducer_weight * transducer_prob
            transducer_weight (float): transducer probability weight in
                prefix beam search
        Returns:
            List[List[int]]: best path result

        """
        self.init_bs()
        beam, _ = self.bs.prefix_beam_search(
            speech,
            speech_lengths,
            decoding_chunk_size,
            beam_size,
            num_decoding_left_chunks,
            simulate_streaming,
            ctc_weight,
            transducer_weight,
        )
        return beam[0].hyp[1:], beam[0].score

    #[AI修改 开始位置 20260115 推理侧Trick1Trick2入口] 作用：Trick1+Trick2 推理侧入口（共享 predictor + ALSD blank 门控）
    def beam_search_hat_aed_alsd(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.0,
        transducer_weight: float = 1.0,
        aed_weight: float = 0.0,
        alsd_blank_threshold: float = 1.0,
        hat_label_weight: float = 1.0,
        aed_attend_window: int = 0,
        max_consecutive_repeats: int = 0,
    ):
        self.init_bs()
        beam, _ = self.bs.prefix_beam_search(
            speech,
            speech_lengths,
            decoding_chunk_size,
            beam_size,
            num_decoding_left_chunks,
            simulate_streaming,
            ctc_weight,
            transducer_weight,
            aed_weight,
            alsd_blank_threshold,
            hat_label_weight,
            aed_attend_window,
            max_consecutive_repeats,
        )
        return beam[0].hyp[1:], beam[0].score
    #[AI修改 结束位置 20260115 推理侧Trick1Trick2入口]

    def transducer_attention_rescoring(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
            reverse_weight: float = 0.0,
            ctc_weight: float = 0.0,
            attn_weight: float = 0.0,
            transducer_weight: float = 0.0,
            search_ctc_weight: float = 1.0,
            search_transducer_weight: float = 0.0,
            beam_search_type: str = 'transducer') -> List[List[int]]:
        """beam search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            ctc_weight (float): ctc probability weight using in rescoring.
                rescore_prob = ctc_weight * ctc_prob +
                               transducer_weight * (transducer_loss * -1) +
                               attn_weight * attn_prob
            attn_weight (float): attn probability weight using in rescoring.
            transducer_weight (float): transducer probability weight using in
                rescoring
            search_ctc_weight (float): ctc weight using
                               in rnnt beam search (seeing in self.beam_search)
            search_transducer_weight (float): transducer weight using
                               in rnnt beam search (seeing in self.beam_search)
        Returns:
            List[List[int]]: best path result

        """

        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        self.init_bs()
        #breakpoint()
        if beam_search_type == 'transducer':
            beam, encoder_out = self.bs.prefix_beam_search(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                beam_size=beam_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                ctc_weight=search_ctc_weight,
                transducer_weight=search_transducer_weight,
            )
            beam_score = [s.score for s in beam]
            hyps = [s.hyp[1:] for s in beam]

        elif beam_search_type == 'ctc':
            hyps, encoder_out = self._ctc_prefix_beam_search(
                speech,
                speech_lengths,
                beam_size=beam_size,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                simulate_streaming=simulate_streaming)
            beam_score = [hyp[1] for hyp in hyps]
            hyps = [hyp[0] for hyp in hyps]
        assert len(hyps) == beam_size

        # build hyps and encoder output
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)

        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)

        # 2.1 calculate transducer score
        td_score = self._cal_transducer_score(
            encoder_out,
            encoder_mask,
            hyps_lens,
            hyps_pad,
        )
        # 2.2 calculate attention score
        decoder_out, r_decoder_out = self._cal_attn_score(
            encoder_out,
            encoder_mask,
            hyps_pad,
            hyps_lens,
        )

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp)][self.eos]
            td_s = td_score[i]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp):
                    r_score += r_decoder_out[i][len(hyp) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp)][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score = score * attn_weight + \
                beam_score[i] * ctc_weight + \
                td_s * transducer_weight
            if score > best_score:
                best_score = score
                best_index = i

        return hyps[best_index], best_score

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        n_steps: int = 64,
    ) -> List[List[int]]:
        """ greedy search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        # TODO(Mddct): batch decode
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # TODO(Mddct): forward chunk by chunk
        _ = simulate_streaming
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps = basic_greedy_search(self,
                                   encoder_out,
                                   encoder_out_lens,
                                   n_steps=n_steps)

        return hyps

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def forward_predictor_step(
            self, xs: torch.Tensor, cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert len(cache) == 2
        # fake padding
        padding = torch.zeros(1, 1)
        return self.predictor.forward_step(xs, padding, cache)

    @torch.jit.export
    def forward_joint_step(self, enc_out: torch.Tensor,
                           pred_out: torch.Tensor) -> torch.Tensor:
        return self.joint(enc_out, pred_out)

    @torch.jit.export
    def forward_predictor_init_state(self) -> List[torch.Tensor]:
        return self.predictor.init_state(1, device=torch.device("cpu"))

    def _compute_loss(self,
                      encoder_out: torch.Tensor,
                      encoder_out_lens: torch.Tensor,
                      encoder_mask: torch.Tensor,
                      text: torch.Tensor,
                      text_lengths: torch.Tensor,
                      steps: int = 0) -> torch.Tensor:
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        # predictor
        predictor_out = self.predictor(ys_in_pad)
        if self.simple_lm_proj is None and self.simple_am_proj is None:
            # joint
            joint_out = self.joint(encoder_out, predictor_out)
            # NOTE(Mddct): some loss implementation require pad valid is zero
            # torch.int32 rnnt_loss required
            rnnt_text = text.to(torch.int64)
            rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                    rnnt_text).to(torch.int32)
            rnnt_text_lengths = text_lengths.to(torch.int32)
            encoder_out_lens = encoder_out_lens.to(torch.int32)
            loss = torchaudio.functional.rnnt_loss(joint_out,
                                                   rnnt_text,
                                                   encoder_out_lens,
                                                   rnnt_text_lengths,
                                                   blank=self.blank,
                                                   reduction="mean")
        else:
            try:
                import k2
            except ImportError:
                print('Error: k2 is not installed')
            delay_penalty = self.delay_penalty
            if steps < 2 * self.warmup_steps:
                delay_penalty = 0.00
            ys_in_pad = ys_in_pad.type(torch.int64)
            boundary = torch.zeros((encoder_out.size(0), 4),
                                   dtype=torch.int64,
                                   device=encoder_out.device)
            boundary[:, 3] = encoder_mask.squeeze(1).sum(1)
            boundary[:, 2] = text_lengths

            rnnt_text = torch.where(text == self.ignore_id, 0, text)
            lm = self.simple_lm_proj(predictor_out)
            am = self.simple_am_proj(encoder_out)
            amp_autocast = torch.cuda.amp.autocast
            if "npu" in self.device.__str__() and TORCH_NPU_AVAILABLE:
                amp_autocast = torch.npu.amp.autocast
            with amp_autocast(enabled=False):
                simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                    lm=lm.float(),
                    am=am.float(),
                    symbols=rnnt_text,
                    termination_symbol=self.blank,
                    lm_only_scale=self.lm_only_scale,
                    am_only_scale=self.am_only_scale,
                    boundary=boundary,
                    reduction="sum",
                    return_grad=True,
                    delay_penalty=delay_penalty,
                )
            # ranges : [B, T, prune_range]
            ranges = k2.get_rnnt_prune_ranges(
                px_grad=px_grad,
                py_grad=py_grad,
                boundary=boundary,
                s_range=5,
            )
            am_pruned, lm_pruned = k2.do_rnnt_pruning(
                am=self.joint.enc_ffn(encoder_out),
                lm=self.joint.pred_ffn(predictor_out),
                ranges=ranges,
            )
            logits = self.joint(
                am_pruned,
                lm_pruned,
                pre_project=False,
            )
            with amp_autocast(enabled=False):
                pruned_loss = k2.rnnt_loss_pruned(
                    logits=logits.float(),
                    symbols=rnnt_text,
                    ranges=ranges,
                    termination_symbol=self.blank,
                    boundary=boundary,
                    reduction="sum",
                    delay_penalty=delay_penalty,
                )
            simple_loss_scale = 0.5
            if steps < self.warmup_steps:
                simple_loss_scale = (1.0 - (steps / self.warmup_steps) *
                                     (1.0 - simple_loss_scale))
            pruned_loss_scale = 1.0
            if steps < self.warmup_steps:
                pruned_loss_scale = 0.1 + 0.9 * (steps / self.warmup_steps)
            loss = (simple_loss_scale * simple_loss +
                    pruned_loss_scale * pruned_loss)
            loss = loss / encoder_out.size(0)
        return loss
