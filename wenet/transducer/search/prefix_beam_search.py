# [AI修改 开始位置] 作用：增加 Optional 类型以支持 ALSD/AED 门控返回
from typing import List, Tuple, Optional
# [AI修改 结束位置]

import math
import torch
from wenet.utils.common import log_add


class Sequence():

    __slots__ = {'hyp', 'score', 'cache'}

    def __init__(
        self,
        hyp: List[torch.Tensor],
        score,
        cache: List[torch.Tensor],
    ):
        self.hyp = hyp
        self.score = score
        self.cache = cache


class PrefixBeamSearch():

    def __init__(self, encoder, predictor, joint, ctc, blank):
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc
        self.blank = blank

    # [AI修改 开始位置] 作用：ALSD 下为 AED 传入 encoder 前缀，并返回逐 beam 门控掩码
    def forward_decoder_one_step(
            self,
            encoder_x: torch.Tensor,
            encoder_prefix: Optional[torch.Tensor],
            pre_t: torch.Tensor,
            cache: List[torch.Tensor],
            aed_weight: float = 0.0,
            alsd_blank_threshold: float = 1.0,
            hat_label_weight: float = 1.0,
            aed_attend_window: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]:
        padding = torch.zeros(pre_t.size(0), 1, device=encoder_x.device)
        pred_out, new_cache = self.predictor.forward_step(pre_t.unsqueeze(-1),
                                                          padding, cache)
        aed_logits = None
        aed_mask = None
        if getattr(self.joint, "is_multi_mode_joiner", False):
            # [AI修改 开始位置] 作用：multi-mode joiner 的 hat 模式已输出 logp，避免再次 log_softmax 破坏分布
            beam_size = pre_t.size(0)
            if encoder_x.size(0) == 1 and beam_size != 1:
                encoder_x = encoder_x.expand(beam_size, -1, -1)
            x = self.joint(encoder_x, pred_out, pre_project=True,
                           mode="hat")  # [beam, 1, 1, vocab]
            if aed_weight > 0.0:
                blank_prob = torch.exp(
                    x.squeeze(1).squeeze(1)[:, self.blank])  # [beam]
                # [AI修改 开始位置] 作用：对齐论文 streaming AED：hat_label_weight=0 时不使用触发阈值，逐帧计算 AED label
                if float(hat_label_weight) == 0.0:
                    aed_mask = torch.ones_like(blank_prob, dtype=torch.bool)
                else:
                    aed_mask = blank_prob <= alsd_blank_threshold  # [beam]
                # [AI修改 结束位置]
                if torch.any(aed_mask):
                    if encoder_prefix is None:
                        encoder_prefix = encoder_x
                    if encoder_prefix.size(0) == 1 and beam_size != 1:
                        encoder_prefix = encoder_prefix.expand(beam_size, -1,
                                                               -1)
                    attn_mask = None
                    if aed_attend_window > 0:
                        t = int(encoder_prefix.size(1))
                        w = int(aed_attend_window)
                        if t > w:
                            attn_mask = torch.zeros((beam_size, 1, t),
                                                    device=encoder_prefix.device,
                                                    dtype=torch.bool)
                            attn_mask[:, :, t - w:] = True
                    aed_logits = self.joint(encoder_prefix,
                                            pred_out,
                                            pre_project=True,
                                            mode="aed",
                                            attn_mask=attn_mask)  # [beam, 1, vocab]
            # [AI修改 结束位置]
        else:
            x = self.joint(encoder_x, pred_out)  # [beam, 1, 1, vocab]
            x = x.log_softmax(dim=-1)
        return x, aed_logits, aed_mask, new_cache
    # [AI修改 结束位置]

    def prefix_beam_search(self,
                           speech: torch.Tensor,
                           speech_lengths: torch.Tensor,
                           decoding_chunk_size: int = -1,
                           beam_size: int = 5,
                           num_decoding_left_chunks: int = -1,
                           simulate_streaming: bool = False,
                           ctc_weight: float = 0.3,
                           transducer_weight: float = 0.7,
                           aed_weight: float = 0.0,
                           alsd_blank_threshold: float = 1.0,
                           hat_label_weight: float = 1.0,
                           aed_attend_window: int = 0,
                           max_consecutive_repeats: int = 0):
        """prefix beam search
           also see wenet.transducer.transducer.beam_search
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        assert batch_size == 1

        # 1. Encoder
        encoder_out, _ = self.encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        if max_consecutive_repeats < 0:
            raise ValueError("max_consecutive_repeats 需 >= 0")

        ctc_probs = self.ctc.log_softmax(encoder_out).squeeze(0)
        beam_init: List[Sequence] = []
        # breakpoint()
        # 2. init beam using Sequence to save beam unit
        cache = self.predictor.init_state(1, method="zero", device=device)
        beam_init.append(Sequence(hyp=[self.blank], score=0.0, cache=cache))
        # 3. start decoding (notice: we use breathwise first searching)
        # !!!! In this decoding method: one frame do not output multi units. !!!!
        # !!!!    Experiments show that this strategy has little impact      !!!!
        for i in range(maxlen):
            # 3.1 building input
            # decoder taking the last token to predict the next token
            input_hyp = [s.hyp[-1] for s in beam_init]
            input_hyp_tensor = torch.tensor(input_hyp,
                                            dtype=torch.int,
                                            device=device)
            # building statement from beam
            cache_batch = self.predictor.cache_to_batch(
                [s.cache for s in beam_init])
            # build score tensor to do torch.add() function
            scores = torch.tensor([s.score for s in beam_init]).to(device)
            # breakpoint()
            # [AI修改 开始位置] 作用：为 AED 提供 encoder 前缀（到当前帧），并接收逐 beam 的 AED 触发掩码
            # 3.2 forward decoder
            logp, aed_logits, aed_mask, new_cache = self.forward_decoder_one_step(
                encoder_out[:, i, :].unsqueeze(1),
                encoder_out[:, :i + 1, :],
                input_hyp_tensor,
                cache_batch,
                aed_weight=aed_weight,
                alsd_blank_threshold=alsd_blank_threshold,
                hat_label_weight=hat_label_weight,
                aed_attend_window=aed_attend_window,
            )  # logp: (N, 1, 1, vocab_size)
            # [AI修改 结束位置]
            logp = logp.squeeze(1).squeeze(1)  # logp: (N, vocab_size)
            new_cache = self.predictor.batch_to_cache(new_cache)

            # 3.3 shallow fusion for transducer score
            #     and ctc score where we can also add the LM score
            if transducer_weight <= 0.0 and ctc_weight <= 0.0:
                raise ValueError("transducer_weight 与 ctc_weight 不能同时为 0")
            # [AI修改 开始位置] 作用：允许 transducer/ctc 任意一侧权重为 0，避免 log(0)
            if transducer_weight <= 0.0:
                logp = ctc_probs[i].unsqueeze(0)
            elif ctc_weight <= 0.0:
                logp = logp
            else:
                logp = torch.log(
                    torch.add(transducer_weight * torch.exp(logp),
                              ctc_weight *
                              torch.exp(ctc_probs[i].unsqueeze(0))))
            # [AI修改 结束位置]

            if hat_label_weight != 1.0:
                # [AI修改 开始位置] 作用：仅保留 HAT 的 blank 约束，弱化/关闭 HAT label 贡献以便单独验证 AED
                if not (0.0 <= hat_label_weight <= 1.0):
                    raise ValueError("hat_label_weight 需在 [0, 1] 范围内")
                blank_logp = logp[:, self.blank:self.blank + 1]
                non_blank_logp = torch.log(
                    torch.clamp(1.0 - torch.exp(blank_logp), min=1e-12))
                label_count = max(logp.size(-1) - 1, 1)
                uniform_label_logp = non_blank_logp - math.log(float(label_count))
                label_mask = torch.ones_like(logp, dtype=torch.bool)
                label_mask[:, self.blank] = False
                logp = torch.where(label_mask,
                                   hat_label_weight * logp +
                                   (1.0 - hat_label_weight) * uniform_label_logp,
                                   logp)
                # [AI修改 结束位置]
            if (aed_logits is not None and aed_mask is not None
                    and torch.any(aed_mask)):
                # [AI修改 开始位置] 作用：避免 HAT+AED 双计分导致插入，按“条件 label 分布”做稳定融合
                if aed_weight < 0.0:
                    raise ValueError("aed_weight 需 >= 0")
                aed_mix = min(float(aed_weight), 1.0)
                blank_logp = logp[:, self.blank:self.blank + 1]
                non_blank_logp = torch.log(
                    torch.clamp(1.0 - torch.exp(blank_logp), min=1e-12))

                label_mask = torch.ones_like(logp, dtype=torch.bool)
                label_mask[:, self.blank] = False
                aed_active = aed_mask.to(device=logp.device,
                                         dtype=torch.bool).unsqueeze(1)
                active_label_mask = label_mask & aed_active
                mask_value = -1e9

                hat_cond = logp - non_blank_logp
                hat_cond = torch.where(label_mask, hat_cond,
                                       hat_cond.new_full(hat_cond.shape, mask_value))
                hat_cond = hat_cond - torch.logsumexp(hat_cond, dim=-1, keepdim=True)

                if hat_label_weight != 1.0:
                    label_count = max(logp.size(-1) - 1, 1)
                    uniform_cond = hat_cond.new_full(hat_cond.shape, mask_value)
                    uniform_cond = torch.where(label_mask,
                                               uniform_cond.new_full(uniform_cond.shape,
                                                                     -math.log(float(label_count))),
                                               uniform_cond)
                    hat_cond = hat_label_weight * hat_cond + (1.0 - hat_label_weight) * uniform_cond
                    hat_cond = torch.where(label_mask, hat_cond,
                                           hat_cond.new_full(hat_cond.shape, mask_value))
                    hat_cond = hat_cond - torch.logsumexp(hat_cond, dim=-1, keepdim=True)

                aed_logp = aed_logits.squeeze(1).log_softmax(dim=-1)
                aed_cond = torch.where(label_mask, aed_logp,
                                       aed_logp.new_full(aed_logp.shape, mask_value))
                aed_cond = aed_cond - torch.logsumexp(aed_cond, dim=-1, keepdim=True)

                comb_cond = (1.0 - aed_mix) * hat_cond + aed_mix * aed_cond
                comb_cond = torch.where(label_mask, comb_cond,
                                        comb_cond.new_full(comb_cond.shape, mask_value))
                comb_cond = comb_cond - torch.logsumexp(comb_cond, dim=-1, keepdim=True)

                fused_logp = torch.where(label_mask, non_blank_logp + comb_cond,
                                         logp)
                logp = torch.where(active_label_mask, fused_logp, logp)
                logp[:, self.blank:self.blank + 1] = blank_logp
                # [AI修改 结束位置]

            if max_consecutive_repeats > 0:
                # [AI修改 开始位置] 作用：抑制连续重复 token，缓解“打打打/退退退”类崩塌
                max_rep = int(max_consecutive_repeats)
                if max_rep < 1:
                    raise ValueError("max_consecutive_repeats 需 >= 0")
                mask_value = -1e9
                for j, s in enumerate(beam_init):
                    last_tok = int(s.hyp[-1])
                    if last_tok == self.blank:
                        continue
                    run_len = 0
                    for tok in reversed(s.hyp):
                        if int(tok) == last_tok:
                            run_len += 1
                        else:
                            break
                    if run_len >= max_rep:
                        logp[j, last_tok] = mask_value
                # [AI修改 结束位置]

            # 3.4 first beam prune
            top_k_logp, top_k_index = logp.topk(beam_size)  # (N, N)
            scores = torch.add(scores.unsqueeze(1), top_k_logp)

            # 3.5 generate new beam (N*N)
            beam_A = []
            for j in range(len(beam_init)):
                # update seq
                base_seq = beam_init[j]
                for t in range(beam_size):
                    # blank: only update the score
                    if top_k_index[j, t] == self.blank:
                        new_seq = Sequence(hyp=base_seq.hyp.copy(),
                                           score=scores[j, t].item(),
                                           cache=base_seq.cache)

                        beam_A.append(new_seq)
                    # other unit: update hyp score statement and last
                    else:
                        hyp_new = base_seq.hyp.copy()
                        hyp_new.append(top_k_index[j, t].item())
                        new_seq = Sequence(hyp=hyp_new,
                                           score=scores[j, t].item(),
                                           cache=new_cache[j])
                        beam_A.append(new_seq)

            # 3.6 prefix fusion
            fusion_A = [beam_A[0]]
            for j in range(1, len(beam_A)):
                s1 = beam_A[j]
                if_do_append = True
                for t in range(len(fusion_A)):
                    # notice: A_ can not fusion with A
                    if s1.hyp == fusion_A[t].hyp:
                        # fusion_A[t].score = log_add(
                        #     [fusion_A[t].score, s1.score])
                        fusion_A[t].score = log_add(
                            fusion_A[t].score, s1.score)
                        if_do_append = False
                        break
                if if_do_append:
                    fusion_A.append(s1)

            # 4. second pruned
            fusion_A.sort(key=lambda x: x.score, reverse=True)
            beam_init = fusion_A[:beam_size]

        return beam_init, encoder_out
