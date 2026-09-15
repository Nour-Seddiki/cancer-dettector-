"""CNN -> Transformer bridge + report decoder.

The attention machinery (`MultiHeadAttention`, `CrossAttention`, `FeedForward`,
`DecoderBlock`) is carried over from the EN->FR project's `transformer/model.py`. The only
change is that `n_emb`/`dropout` are constructor arguments instead of module globals, since
this project runs different hyperparameters.

Nothing in `DecoderBlock` had to change to make this work. That is the point of
PROJECT_PLAN.md Section 3: the decoder cross-attends over "a sequence of vectors" and does not
care whether that sequence came from a text encoder or from a CNN's flattened 7x7 grid.
`VisualBridge` is the entire adapter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_model import FEATURE_DIM, NUM_REGIONS, DenseNet121Encoder
from labels import NO_FINDING_IDX

# Decoder hyperparameters. Smaller than the translation model on purpose: the vocab is
# ~1.5k tokens instead of ~100k, and IU X-Ray has ~3k training studies, so a 6-layer/384-dim
# stack would overfit long before it converged. 3x512 matches the R2Gen reference setup.
n_emb = 512
n_head = 8
n_layer = 3
dropout = 0.1
block_size = 160
label_smoothing = 0.1
IGNORE_INDEX = -100


class CrossAttention(nn.Module):
    def __init__(self, head_size, n_emb, n_head, dropout=dropout):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.q_proj = nn.Linear(n_emb, head_size * n_head, bias=False)
        self.kv_proj = nn.Linear(n_emb, 2 * head_size * n_head, bias=False)
        self.out_proj = nn.Linear(head_size * n_head, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        B, T, _ = x.shape
        S = context.shape[1]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        kv = self.kv_proj(context).view(B, S, 2, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)  # fused kernel, no mask (full context)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)
        return self.dropout(self.out_proj(out))


class MultiHeadAttention(nn.Module):
    """Causal self-attention over the report tokens.

    No padding mask, deliberately: padding only ever sits at the *end* of a sequence, so
    under a causal mask a real token never attends to a pad, and the pad positions' own
    outputs are dropped from the loss by `ignore_index`.
    """

    def __init__(self, head_size, n_head, n_emb, causal=True, dropout=dropout):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.causal = causal
        self.qkv = nn.Linear(n_emb, 3 * head_size * n_head, bias=False)
        self.proj = nn.Linear(head_size * n_head, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # is_causal on a length-1 query (the incremental decode path) is a no-op anyway.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and T > 1)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_emb, dropout=dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_emb, n_head, dropout=dropout):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(head_size, n_head, n_emb, causal=True, dropout=dropout)
        self.ca = CrossAttention(head_size, n_emb, n_head, dropout=dropout)
        self.ffd = FeedForward(n_emb, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln_ca = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x, context):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln_ca(x), context)
        x = x + self.ffd(self.ln2(x))
        return x


class VisualBridge(nn.Module):
    """(B, 49, 1024) CNN tokens -> (B, 49, n_emb) decoder context.

    Projection + LayerNorm + a learned positional embedding over the 49 grid cells, so the
    decoder can tell an apex from a costophrenic angle. A row/col-factored embedding is the
    stretch-goal variant noted in the plan; `factorized=True` enables it.
    """

    def __init__(self, feature_dim=FEATURE_DIM, n_emb=n_emb, num_regions=NUM_REGIONS,
                 dropout=dropout, factorized=False):
        super().__init__()
        self.num_regions = num_regions
        self.factorized = factorized
        self.proj = nn.Linear(feature_dim, n_emb)
        self.ln = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(dropout)

        if factorized:
            grid = int(num_regions ** 0.5)
            assert grid * grid == num_regions, "factorized positions need a square grid"
            self.grid = grid
            self.row_emb = nn.Embedding(grid, n_emb)
            self.col_emb = nn.Embedding(grid, n_emb)
        else:
            self.pos_emb = nn.Embedding(num_regions, n_emb)

    def forward(self, features):
        B, S, _ = features.shape
        x = self.proj(features)
        idx = torch.arange(S, device=features.device)
        if self.factorized:
            x = x + self.row_emb(idx // self.grid) + self.col_emb(idx % self.grid)
        else:
            x = x + self.pos_emb(idx)
        return self.drop(self.ln(x))


class LabelBridge(nn.Module):
    """(B, 14) condition probabilities -> (B, 14, n_emb) extra decoder context.

    Token c interpolates between learned "c present" and "c absent" embeddings by the
    probability the CNN's classifier head gives c. Appended to the 49 image tokens, it hands
    the decoder the classifier's verdict in a form it can cross-attend to directly. Without
    it the decoder has to rediscover every finding from raw features, and on ~3k studies it
    mostly does not: the stage (a) classifier reached AUROC ~0.9 on cardiomegaly and
    effusion while the decoder, over the same features, almost never wrote either.
    """

    def __init__(self, num_conditions, n_emb=n_emb, dropout=dropout):
        super().__init__()
        self.present = nn.Parameter(torch.randn(num_conditions, n_emb) * 0.02)
        self.absent = nn.Parameter(torch.randn(num_conditions, n_emb) * 0.02)
        self.ln = nn.LayerNorm(n_emb)
        self.drop = nn.Dropout(dropout)

    def forward(self, probs):
        p = probs.unsqueeze(-1)
        return self.drop(self.ln(p * self.present + (1 - p) * self.absent))


class ReportGenerator(nn.Module):
    """DenseNet121 -> VisualBridge (+ LabelBridge) -> Transformer decoder -> report tokens."""

    # (14,) per-condition thresholds, set by evaluate.load_generator when the checkpoint
    # carries tuned ones (tune_thresholds.py). None = soft probability condition tokens.
    label_thresholds = None

    def __init__(self, vocab_size, pad_id=0, n_emb=n_emb, n_head=n_head, n_layer=n_layer,
                 block_size=block_size, dropout=dropout, pretrained_cnn=True,
                 encoder=None, factorized_pos=False, label_tokens=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.block_size = block_size

        self.encoder = encoder if encoder is not None else DenseNet121Encoder(pretrained=pretrained_cnn)
        self.bridge = VisualBridge(n_emb=n_emb, dropout=dropout, factorized=factorized_pos)
        self.label_bridge = (LabelBridge(self.encoder.classifier.out_features, n_emb, dropout)
                             if label_tokens else None)

        self.tok_emb = nn.Embedding(vocab_size, n_emb, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(block_size, n_emb)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(DecoderBlock(n_emb, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        # Weight tying: the vocab is small but the paired data is smaller, and tying cuts
        # ~0.8M free parameters that would otherwise only see a handful of updates each.
        self.lm_head.weight = self.tok_emb.weight

        # Everything except the encoder: `self.apply` would also hit the encoder's Linear
        # classifier head and silently wipe the stage (a) weights it was warm-started with.
        for name, module in self.named_children():
            if name != "encoder":
                module.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- forward -----------------------------------------------------------------------

    def encode(self, images, teacher_labels=None, teacher_prob=0.0):
        """Images -> ((B, 49 [+14], n_emb) cross-attention context, (B, 14) condition logits).

        `teacher_labels` / `teacher_prob` are for training only: with probability
        `teacher_prob` per study, the condition tokens are built from the ground-truth labels
        instead of the head's predictions. Trained purely on its own noisy predictions, the
        decoder learns to half-ignore the tokens and the normal-template prior still wins;
        seeing the true labels part of the time makes them a signal it has to follow, and
        the rest of the time keeps it used to the head's soft outputs it gets at test time.
        """
        feats = self.encoder.forward_features(images)            # (B, 49, 1024)
        cls_logits = self.encoder.classifier(feats.mean(dim=1))  # same pooling as stage (a)
        context = self.bridge(feats)
        if self.label_bridge is not None:
            # Detached: the tokens carry what the classifier head believes (trained only by
            # the auxiliary BCE), not a free code the LM loss could repurpose.
            probs = torch.sigmoid(cls_logits.float()).detach()
            if teacher_labels is not None and teacher_prob > 0:
                use = torch.rand(probs.shape[0], 1, device=probs.device) < teacher_prob
                probs = torch.where(use, teacher_labels.float(), probs)
            elif self.label_thresholds is not None:
                # Binary tokens at per-condition thresholds tuned on val (tune_thresholds.py).
                # A decoder trained on ground-truth 0/1 tokens follows a confident 0/1 far
                # more reliably than a soft probability it never saw during training.
                thr = self.label_thresholds
                probs = (probs >= thr).float()
                if torch.isnan(thr[NO_FINDING_IDX]):
                    # NaN = derive "No Finding" (tune_thresholds.py --no-finding derived): it
                    # means "no abnormal condition" in every ground-truth label the decoder
                    # saw, so set it from the abnormal tokens instead of thresholding it alone.
                    abnormal = torch.cat([probs[:, :NO_FINDING_IDX],
                                          probs[:, NO_FINDING_IDX + 1:]], dim=1)
                    probs[:, NO_FINDING_IDX] = 1.0 - abnormal.amax(dim=1)
            context = torch.cat([context, self.label_bridge(probs).to(context.dtype)], dim=1)
        return context, cls_logits

    def encode_image(self, images):
        """Images -> cross-attention context."""
        return self.encode(images)[0]

    def decode(self, tgt_ids, context):
        B, T = tgt_ids.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(T, device=tgt_ids.device)
        x = self.drop(self.tok_emb(tgt_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, context)
        return self.lm_head(self.ln_f(x))

    def forward(self, images, tgt_in, targets=None, context=None):
        context = self.encode_image(images) if context is None else context
        logits = self.decode(tgt_in, context)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
        )
        return logits, loss

    # -- fine-tuning control -------------------------------------------------------------

    def freeze_cnn(self):
        self.encoder.freeze_backbone()
        return self

    def unfreeze_cnn_last_block(self):
        self.encoder.unfreeze_last_block()
        return self

    def param_groups(self, lr, cnn_lr):
        """Discriminative LRs: CNN << decoder (plan Section 3)."""
        cnn_params = [p for p in self.encoder.parameters() if p.requires_grad]
        cnn_ids = {id(p) for p in self.encoder.parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in cnn_ids]
        groups = [{"params": rest, "lr": lr}]
        if cnn_params:
            groups.append({"params": cnn_params, "lr": cnn_lr})
        return groups

    # -- generation ----------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, images, max_new_tokens, bos_token_id, eos_token_id=None,
                 beam_size=1, length_penalty=0.6, context=None):
        context = self.encode_image(images) if context is None else context
        if beam_size > 1:
            return self._generate_beam(context, max_new_tokens, bos_token_id,
                                       eos_token_id, beam_size, length_penalty)

        B = context.shape[0]
        device = context.device
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode(generated[:, -self.block_size:], context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token[finished] = eos_token_id
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and finished.all():
                break
        return generated[:, 1:]

    @torch.no_grad()
    def _generate_beam(self, context, max_new_tokens, bos_token_id, eos_token_id,
                       beam_size, length_penalty):
        """Batched beam search, ranked by length-normalised cumulative log-prob.

        Same implementation as `transformer/model.py`; the only difference is that the
        context being expanded across beams is the 49 image tokens rather than an encoded
        source sentence.
        """
        device = context.device
        B, S, C = context.shape
        k = beam_size
        eos = eos_token_id if eos_token_id is not None else -1

        context = context.unsqueeze(1).expand(B, k, S, C).reshape(B * k, S, C)
        beams = torch.full((B * k, 1), bos_token_id, dtype=torch.long, device=device)
        # Only beam 0 starts active per image, so step 1 does not clone the top token k times.
        scores = torch.full((B, k), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.reshape(-1)
        lengths = torch.zeros(B * k, device=device)
        finished = torch.zeros(B * k, dtype=torch.bool, device=device)
        batch_offset = (torch.arange(B, device=device) * k).unsqueeze(-1)

        for _ in range(max_new_tokens):
            logits = self.decode(beams[:, -self.block_size:], context)
            log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
            V = log_probs.shape[-1]

            if finished.any():
                # A finished beam may only continue by re-emitting eos at no score cost.
                log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
                log_probs[finished, eos] = 0.0

            candidates = scores.unsqueeze(-1) + log_probs
            candidate_lengths = (lengths + (~finished).float()).clamp(min=1)
            norm = candidates / candidate_lengths.unsqueeze(-1).pow(length_penalty)

            _, top_idx = norm.reshape(B, k * V).topk(k, dim=-1)
            beam_idx = torch.div(top_idx, V, rounding_mode="floor")
            token_idx = (top_idx % V).reshape(-1)
            flat = (beam_idx + batch_offset).reshape(-1)

            scores = candidates[flat, token_idx]
            lengths = lengths[flat] + (~finished[flat]).float()
            finished = finished[flat] | (token_idx == eos)
            beams = torch.cat([beams[flat], token_idx.reshape(-1, 1)], dim=1)

            if finished.all():
                break

        final = (scores / lengths.clamp(min=1).pow(length_penalty)).reshape(B, k)
        best = final.argmax(dim=-1)
        best_beams = beams.reshape(B, k, -1)[torch.arange(B, device=device), best]
        return best_beams[:, 1:]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 1500
    model = ReportGenerator(vocab_size=V, pad_id=0, pretrained_cnn=False).to(device)

    images = torch.randn(2, 3, 224, 224, device=device)
    tgt_in = torch.randint(4, V, (2, 20), device=device)
    targets = torch.randint(4, V, (2, 20), device=device)

    logits, loss = model(images, tgt_in, targets)
    print(f"logits {tuple(logits.shape)}  loss {loss.item():.3f} "
          f"(expect ~{torch.log(torch.tensor(float(V))).item():.2f} at init)")

    total, trainable = count_parameters(model)
    print(f"params {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    model.eval()
    greedy = model.generate(images, 12, bos_token_id=1, eos_token_id=2)
    beam = model.generate(images, 12, bos_token_id=1, eos_token_id=2, beam_size=3)
    print(f"greedy {tuple(greedy.shape)}  beam {tuple(beam.shape)}")
