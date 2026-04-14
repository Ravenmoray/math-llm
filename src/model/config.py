from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Config explored: 454M / 742M / 1.06B / 1.5B all fit in 16GB with bf16 +
    # AdamW8bit + grad-ckpt at ctx=2048, micro-batch=2.
    # Practical sweet spot given corpus size (~40M tokens) and single-GPU compute
    # budget is ~1B — bigger is data-starved. See notes/training.md.
    vocab_size: int = 32000
    n_layer: int = 30
    n_head: int = 28
    n_kv_head: int = 4        # GQA group size 7
    d_model: int = 1792
    ffn_dim: int = 4864
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    def param_count(self) -> int:
        d = self.d_model
        hd = self.head_dim
        per_layer = (
            # attention
            d * d +                    # Wq
            d * self.n_kv_head * hd +  # Wk
            d * self.n_kv_head * hd +  # Wv
            d * d +                    # Wo
            # ffn (SwiGLU: 3 matrices)
            d * self.ffn_dim * 3 +
            # norms (small)
            2 * d
        )
        total = per_layer * self.n_layer
        # embeddings (tied with lm_head so count once)
        total += self.vocab_size * d
        # final norm
        total += d
        if not self.tie_embeddings:
            total += self.vocab_size * d
        return total
