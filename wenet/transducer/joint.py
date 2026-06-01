from typing import Optional

import math
import torch
from torch import nn
import torch.nn.functional as F
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


class TransducerJoint(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 enc_output_size: int,
                 pred_output_size: int,
                 join_dim: int,
                 prejoin_linear: bool = True,
                 postjoin_linear: bool = False,
                 joint_mode: str = 'add',
                 activation: str = "tanh",
                 hat_joint: bool = False,
                 dropout_rate: float = 0.1,
                 hat_activation: str = 'tanh'):
        # TODO(Mddct): concat in future
        assert joint_mode in ['add']
        super().__init__()

        self.activatoin = WENET_ACTIVATION_CLASSES[activation]()
        self.prejoin_linear = prejoin_linear
        self.postjoin_linear = postjoin_linear
        self.joint_mode = joint_mode

        if not self.prejoin_linear and not self.postjoin_linear:
            assert enc_output_size == pred_output_size == join_dim
        # torchscript compatibility
        self.enc_ffn: Optional[nn.Linear] = None
        self.pred_ffn: Optional[nn.Linear] = None
        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)
        # torchscript compatibility
        self.post_ffn: Optional[nn.Linear] = None
        if self.postjoin_linear:
            self.post_ffn = nn.Linear(join_dim, join_dim)

        # NOTE: <blank> in vocab_size
        self.hat_joint = hat_joint
        self.vocab_size = vocab_size
        self.ffn_out: Optional[torch.nn.Linear] = None
        if not self.hat_joint:
            self.ffn_out = nn.Linear(join_dim, vocab_size)

        self.blank_pred: Optional[torch.nn.Module] = None
        self.token_pred: Optional[torch.nn.Module] = None
        if self.hat_joint:
            self.blank_pred = torch.nn.Sequential(
                torch.nn.Tanh(), torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(join_dim, 1), torch.nn.LogSigmoid())
            self.token_pred = torch.nn.Sequential(
                WENET_ACTIVATION_CLASSES[hat_activation](),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(join_dim, self.vocab_size - 1))

    def forward(self,
                enc_out: torch.Tensor,
                pred_out: torch.Tensor,
                pre_project: bool = True) -> torch.Tensor:
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        if (pre_project and self.prejoin_linear and self.enc_ffn is not None
                and self.pred_ffn is not None):
            enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,D]
            pred_out = self.pred_ffn(pred_out)
        if enc_out.ndim != 4:
            enc_out = enc_out.unsqueeze(2)  # [B,T,D] -> [B,T,1,D]
        if pred_out.ndim != 4:
            pred_out = pred_out.unsqueeze(1)  # [B,U,D] -> [B,1,U,D]

        # TODO(Mddct): concat joint
        _ = self.joint_mode
        out = enc_out + pred_out  # [B,T,U,V]

        if self.postjoin_linear and self.post_ffn is not None:
            out = self.post_ffn(out)

        if not self.hat_joint and self.ffn_out is not None:
            out = self.activatoin(out)
            out = self.ffn_out(out)
            return out
        else:
            assert self.blank_pred is not None
            assert self.token_pred is not None
            blank_logp = self.blank_pred(out)  # [B,T,U,1]

            # scale blank logp
            scale_logp = torch.clamp(1 - torch.exp(blank_logp), min=1e-6)
            label_logp = self.token_pred(out).log_softmax(
                dim=-1)  # [B,T,U,vocab-1]
            # scale token logp
            label_logp = torch.log(scale_logp) + label_logp

            out = torch.cat((blank_logp, label_logp), dim=-1)  # [B,T,U,vocab]
            return out


class MultiModeJoiner(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 enc_output_size: int,
                 pred_output_size: int,
                 join_dim: int,
                 num_heads: int = 8,
                 ff_hidden: int = 2048,
                 dropout_rate: float = 0.1,
                 activation: str = "tanh",
                 blank_id: int = 0,
                 eps: float = 1e-6):
        """
        多模式联合器（Multi-Mode Joiner），用于在同一套参数下模拟多种 ASR 模式的输出。

        设计目标：
        - 在 Transducer/HAT、AED、CTC、LM 四种行为间切换时，共享 Encoder 与 Predictor 的参数；
        - Joiner 内部通过不同的注意力/置零策略实现模式切换，避免多分支 decoder 的重复计算开销。

        对应论文（All-in-One ASR）要点：
        - HAT：sigmoid cross-attention（论文 II-A，Eq.(8)~(16)），逐帧可计算，适用于流式帧同步解码
        - AED：softmax cross-attention（论文 II-B，Eq.(22)~(26)），得到 label 概率（不输出 blank）
        - CTC：将 predictor 侧置零（论文 II-C，Eq.(31)~(38)），使 sigmoid attention 退化为常数 0.5
        - LM：将 encoder 侧置零（论文 II-D，Eq.(39)~(42)），仅依赖 predictor（内部 LM）

        约定：
        - vocab_size 包含 blank（blank_id 指定其在词表中的位置）
        - enc_out: [B, T, enc_output_size]
        - pred_out: [B, U, pred_output_size]（注意：这里的 pred_out 是 predictor 的输出特征，不是 token id）
        - 本实现中：
          - HAT/CTC 返回 log 概率（logp），供 RNNT/HAT/CTC loss 直接使用
          - AED/LM 返回 logits（未 softmax），供 CrossEntropyLoss 使用
        """
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if join_dim % num_heads != 0:
            raise ValueError("join_dim must be divisible by num_heads")
        if blank_id < 0 or blank_id >= vocab_size:
            raise ValueError("invalid blank_id")

        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_heads = num_heads
        self.head_dim = join_dim // num_heads
        self.join_dim = join_dim
        # 记录 predictor 输出维度，用于在 CTC 模式下构造“空 predictor 输入”
        self.pred_input_dim = pred_output_size
        # eps 用于避免 log(0) 或数值下溢导致 NaN
        #[AI修改 开始位置 20260115 eps类型兼容] 作用：强制将 eps 转换为 float，防止配置文件读取为 string 导致 torch.clamp 报错
        self.eps = float(eps)
        #[AI修改 结束位置 20260115 eps类型兼容]

        # 统一投影到 joiner 空间（join_dim）
        self.enc_ffn = nn.Linear(enc_output_size, join_dim)
        self.pred_ffn = nn.Linear(pred_output_size, join_dim)

        # 论文中 joiner 采用 LayerNorm 处理 Q / K,V / FFN 输出
        self.ln_q = nn.LayerNorm(join_dim)
        self.ln_kv = nn.LayerNorm(join_dim)
        self.ln_ff = nn.LayerNorm(join_dim)

        # Q/K/V 与投影层：用于 cross-attention（HAT/AED 模式）
        self.w_query = nn.Linear(join_dim, join_dim)
        self.w_key = nn.Linear(join_dim, join_dim)
        self.w_value = nn.Linear(join_dim, join_dim)
        self.w_proj = nn.Linear(join_dim, join_dim)

        # FFN：用于提升 joiner 表达能力（论文中用于 label 分支）
        act = WENET_ACTIVATION_CLASSES[activation]()
        self.ff = nn.Sequential(
            nn.Linear(join_dim, ff_hidden),
            act,
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden, join_dim),
            nn.Dropout(dropout_rate),
        )

        # blank 分支：输出 blank 的 logit（后续用 logsigmoid 得到 logp）
        self.blank_pred = nn.Linear(join_dim, 1)
        # label 分支：不直接预测 blank，因此维度为 vocab_size-1
        self.label_pred = nn.Linear(join_dim, vocab_size - 1)
        # CTC loss：这里复用 PyTorch 原生 CTCLoss，对应论文 CTC mode 的训练目标
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction="sum",
                                         zero_infinity=True)

        # 用于在上层快速判断是否启用多模式训练路径
        self.is_multi_mode_joiner = True

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, T, H, Dh]
        b, t, d = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, Dh] -> [B, T, D]
        b, t, h, d = x.shape
        return x.contiguous().view(b, t, h * d)

    def _pack_full_vocab_logits(self, label_logits: torch.Tensor) -> torch.Tensor:
        # AED/LM 模式只输出 label（不含 blank），这里将 blank 位置补为一个极小值，得到 [*, vocab_size]
        # 注意：这里返回的是 logits（未 softmax），便于用 CrossEntropyLoss 训练。
        # 这里用 -1e9 而不是 -inf，避免在某些设备/AMP 场景下产生 NaN。
        if self.blank_id == 0:
            blank = label_logits.new_full(label_logits.shape[:-1] + (1, ),
                                          -1e9)
            return torch.cat([blank, label_logits], dim=-1)
        left = label_logits[..., :self.blank_id]
        right = label_logits[..., self.blank_id:]
        blank = label_logits.new_full(label_logits.shape[:-1] + (1, ), -1e9)
        return torch.cat([left, blank, right], dim=-1)

    def _hat_log_probs(self, hjoiner: torch.Tensor) -> torch.Tensor:
        """
        HAT/CTC 模式输出的 log 概率（与 torchaudio.rnnt_loss 兼容）。

        输出形式：
        - blank 概率：logsigmoid(blank_logit)
        - label 概率：softmax(label_logits) 并整体乘以 (1 - P(blank))
        - 最终返回 [*, vocab_size] 的 logp（包含 blank）
        """
        blank_logp = F.logsigmoid(self.blank_pred(hjoiner))
        label_logits = self.label_pred(self.ln_ff(self.ff(hjoiner) + hjoiner))
        label_logp = F.log_softmax(label_logits, dim=-1)
        # 将 label 概率整体乘上 (1 - P(blank))，并在 log 空间相加
        scale_logp = torch.log(torch.clamp(1.0 - torch.exp(blank_logp),
                                           min=self.eps))
        label_logp = scale_logp + label_logp
        if self.blank_id == 0:
            return torch.cat([blank_logp, label_logp], dim=-1)
        left = label_logp[..., :self.blank_id]
        right = label_logp[..., self.blank_id:]
        return torch.cat([left, blank_logp, right], dim=-1)

    def _sigmoid_cross_attention(self, enc: torch.Tensor,
                                 pred: torch.Tensor) -> torch.Tensor:
        """
        HAT 模式的 sigmoid cross-attention（逐帧可计算）。

        输入：
        - enc: [B, T, D]（已经投影到 join_dim）
        - pred: [B, U, D]（已经投影到 join_dim）

        输出：
        - ctx: [B, T, U, D]

        说明：
        - 这里使用“逐元素乘积 + sigmoid”生成每个 head 的权重（论文中的 sigmoid attention）
        - 不做沿时间维的归一化（区别于 softmax attention），利于帧同步
        - 变量对齐论文（II-A）：
          - henc'  / hpred' : enc / pred（已在 join_dim 空间）
          - q_u, k_t, v_t : q / k / v（按 head 切分）
          - α(HAT) 与 c(HAT) : alpha 与 ctx（注意：本实现未做跨时间归一化）
        """
        # q: [B, U, H, Dh]
        q = self._split_heads(self.w_query(self.ln_q(pred)))
        kv_in = self.ln_kv(enc)
        # k/v: [B, T, H, Dh]
        k = self._split_heads(self.w_key(kv_in))
        v = self._split_heads(self.w_value(kv_in))

        # scores: [B, T, U, H]
        scores = (q.unsqueeze(1) * k.unsqueeze(2)).sum(-1) / math.sqrt(
            float(self.head_dim))
        # alpha: [B, T, U, H]，逐帧 sigmoid 注意力权重
        alpha = torch.sigmoid(scores)
        # ctx: [B, T, U, H, Dh]
        ctx = alpha.unsqueeze(-1) * v.unsqueeze(2)
        ctx = ctx.contiguous().view(ctx.size(0), ctx.size(1), ctx.size(2),
                                    self.join_dim)
        return self.w_proj(ctx)  # [B,T,U,D]

    def _softmax_cross_attention(self,
                                 enc: torch.Tensor,
                                 pred: torch.Tensor,
                                 attn_mask: Optional[torch.Tensor] = None
                                 ) -> torch.Tensor:
        """
        AED 模式的 softmax cross-attention（类似单层 Transformer decoder 的 cross-attn）。

        输入：
        - enc: [B, T, D]
        - pred: [B, U, D]
        - attn_mask: [B, U, T]，True 表示可 attend，False 表示被 mask

        输出：
        - ctx: [B, U, D]

        变量对齐论文（II-B）：
        - q_u : 由 pred 生成的 query
        - K/V : 由 enc 生成的 key/value 矩阵
        - α(AED) : softmax 注意力权重（沿时间维归一化）
        """
        # q: [B, U, H, Dh]
        q = self._split_heads(self.w_query(self.ln_q(pred)))
        kv_in = self.ln_kv(enc)
        # k/v: [B, T, H, Dh]
        k = self._split_heads(self.w_key(kv_in))
        v = self._split_heads(self.w_value(kv_in))

        # scores: [B, U, T, H]
        scores = (q.unsqueeze(2) * k.unsqueeze(1)).sum(-1) / math.sqrt(
            float(self.head_dim))
        # -> [B, U, H, T]
        scores = scores.permute(0, 1, 3, 2)
        if attn_mask is not None:
            # attn_mask: [B, U, T]，False 的位置不允许 attend
            scores = scores.masked_fill(~attn_mask.unsqueeze(2), -1e9)
        # alpha: [B, U, H, T]
        alpha = F.softmax(scores, dim=-1)

        #[AI修改 开始位置 20260115 修正AED模式维度广播] 作用：修正 AED 模式下 cross-attention 的维度广播错误
        # v: [B, T, H, Dh]
        # alpha: [B, U, H, T]
        # 目标：ctx = alpha * v (在 T 维度求和)
        # 调整 v 的维度以匹配 alpha: [B, T, H, Dh] -> [B, 1, H, T, Dh]
        v = v.permute(0, 2, 1, 3).unsqueeze(1)  # [B, H, T, Dh] -> [B, 1, H, T, Dh]
        # ctx: [B, U, H, T, 1] * [B, 1, H, T, Dh] -> sum(-2) -> [B, U, H, Dh]
        ctx = (alpha.unsqueeze(-1) * v).sum(-2)
        #[AI修改 结束位置 20260115 修正AED模式维度广播]

        # ctx: [B, U, D]
        ctx = self._merge_heads(ctx)
        return self.w_proj(ctx)

    def forward(self,
                enc_out: Optional[torch.Tensor],
                pred_out: torch.Tensor,
                pre_project: bool = True,
                mode: str = "hat",
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        根据 mode 计算不同模式的输出。

        mode 说明：
        - "hat": 返回 [B, T, U, V] 的 logp（包含 blank），用于 RNNT/HAT loss
        - "aed": 返回 [B, U, V] 的 logits（包含 blank 的占位），用于 CE loss
        - "ctc": 返回 [B, T, V] 的 logp（包含 blank），用于 CTC loss
        - "lm" : 返回 [B, U, V] 的 logits（包含 blank 的占位），用于内部 LM 的 CE loss

        pre_project：
        - True：输入 enc_out/pred_out 仍是原始维度，内部会通过 enc_ffn/pred_ffn 投影到 join_dim
        - False：输入已经在 join_dim 空间（用于某些剪枝或预投影路径）
        """
        mode = str(mode).lower()
        if mode not in ("hat", "aed", "ctc", "lm"):
            raise ValueError("unsupported mode: {}".format(mode))

        if mode == "lm":
            # LM 模式：不使用 encoder，等价于将 K/V 置零，仅依赖 predictor 侧的表示
            if pre_project:
                pred = self.pred_ffn(pred_out)
            else:
                pred = pred_out
            h = torch.tanh(pred)
            label_logits = self.label_pred(self.ln_ff(self.ff(h) + h))
            return self._pack_full_vocab_logits(label_logits)

        if enc_out is None:
            raise ValueError("enc_out is required for mode {}".format(mode))

        if pre_project:
            # 统一投影到 join_dim 空间，便于后续 attention 与 FFN 共享参数
            enc = self.enc_ffn(enc_out)
            pred = self.pred_ffn(pred_out)
        else:
            enc = enc_out
            pred = pred_out

        if mode == "ctc":
            # CTC 模式：将 predictor 侧置零，使 sigmoid attention 退化为常数 0.5
            kv_in = self.ln_kv(enc)
            v = self._split_heads(self.w_value(kv_in))  # [B,T,H,Dh]
            ctx = 0.5 * v
            ctx = self.w_proj(self._merge_heads(ctx))  # [B,T,D]
            h = torch.tanh(ctx)
            return self._hat_log_probs(h)

        if mode == "aed":
            # AED 模式：softmax cross-attention 聚合 encoder，再输出 label logits
            ctx = self._softmax_cross_attention(enc, pred, attn_mask=attn_mask)
            h = torch.tanh(pred + ctx)
            label_logits = self.label_pred(self.ln_ff(self.ff(h) + h))
            return self._pack_full_vocab_logits(label_logits)

        # HAT 模式：sigmoid cross-attention，输出 [B,T,U,V] 的 logp（用于 RNNT/HAT loss）
        ctx = self._sigmoid_cross_attention(enc, pred)
        h = torch.tanh(pred.unsqueeze(1) + ctx)
        return self._hat_log_probs(h)

    def ctc_forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        便捷接口：只给 enc_out 得到 CTC 模式的 logp。
        """
        dummy_pred = enc_out.new_zeros((enc_out.size(0), 1, self.pred_input_dim))
        logp = self.forward(enc_out, dummy_pred, pre_project=True, mode="ctc")
        return logp

    def ctc_loss_fn(self, enc_out: torch.Tensor, hlens: torch.Tensor,
                    ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """
        计算 CTC 模式的 loss。

        说明：
        - 使用 torch.nn.CTCLoss，期望输入为 log 概率（log_softmax 后）
        - ys_pad 中不应包含 ignore_id，调用方需提前把 padding 替换为 blank_id 或裁剪掉
        """
        logp = self.ctc_forward(enc_out)  # [B,T,V]
        logp = logp.transpose(0, 1)  # [T,B,V]
        loss = self.ctc_loss(logp, ys_pad, hlens, ys_lens)
        loss = loss / logp.size(1)
        return loss
