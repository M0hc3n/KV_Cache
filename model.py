import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for the queries
    n_kv_heads: Optional[int] = None  # Number of heads for the keys and values
    vocab_size: int = -1  # This will be set during tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cuda"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula theta_i = 10000^(-2(i-1)/d) for i = [1,2,...,d/2]
    # Shape: (Head_dim/2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Create position indices
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # Convert to complex exponential form: e^(i * m * theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, Head_dim) -> (B, seq_len, H, Head_dim/2, 2)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # (B, seq_len, H, Head_dim/2, 2) -> (B, seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x_reshaped)
    # (seq_len, Head_dim/2) -> (1, seq_len, 1, Head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Apply rotation
    x_rotated = x_complex * freqs_complex
    # Convert back to real
    x_out = torch.view_as_real(x_rotated)
    # Reshape back to original shape
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_repeats: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_repeats == 1:
        return x
    return (
        # (B, seq_len, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # (B, seq_len, n_kv_heads, n_repeats, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_repeats, head_dim)
        # (B, seq_len, n_kv_heads * n_repeats, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_repeats, head_dim)
    )


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(args.dim, hidden_dim, bias=False)  # Up projection
        self.w3 = nn.Linear(hidden_dim, args.dim, bias=False)  # Down projection

    def forward(self, x: torch.Tensor):
        # SwiGLU implementation: SwiGLU(x, W, V, W2) = (Swish(xW) ⊙ xV)W2
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicate the number of keys and values heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicate the number of heads for the queries
        self.n_q_heads = args.n_heads
        # Indicate how many times the keys and values should be repeated
        self.n_rep = self.n_q_heads // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Initialize cache tensors
        self.register_buffer(
            "cache_keys",
            torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
            ),
        )
        self.register_buffer(
            "cache_values",
            torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
            ),
        )

    @staticmethod
    def attention(
        query, key, value, head_dim: int, mask: Optional[torch.Tensor] = None
    ):
        # Compute attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask if provided
        if mask is not None:
            attention_scores = attention_scores + mask

        attention_scores = F.softmax(attention_scores, dim=-1).type_as(query)
        output = attention_scores @ value
        return output

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, seq_len, Dim)

        # Apply linear transformations
        # (B, seq_len, Dim) -> (B, seq_len, H_Q * Head_Dim)
        query = self.wq(x)
        # (B, seq_len, Dim) -> (B, seq_len, H_KV * Head_Dim)
        key = self.wk(x)
        # (B, seq_len, Dim) -> (B, seq_len, H_KV * Head_Dim)
        value = self.wv(x)

        # Reshape for multi-head attention
        # (B, seq_len, H_Q * Head_Dim) -> (B, seq_len, H_Q, Head_Dim)
        query = query.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, seq_len, H_KV * Head_Dim) -> (B, seq_len, H_KV, Head_Dim)
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, seq_len, H_KV * Head_Dim) -> (B, seq_len, H_KV, Head_Dim)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        # Extract the relevant frequencies for the current positions
        freqs_complex_seq = freqs_complex[start_pos : start_pos + seq_len]
        query = apply_rotary_embeddings(query, freqs_complex_seq, device=x.device)
        key = apply_rotary_embeddings(key, freqs_complex_seq, device=x.device)

        # Update KV cache
        self.cache_keys[:batch_size, start_pos : start_pos + seq_len] = key
        self.cache_values[:batch_size, start_pos : start_pos + seq_len] = value

        # Retrieve keys and values from cache
        keys = self.cache_keys[:batch_size, : start_pos + seq_len]
        values = self.cache_values[:batch_size, : start_pos + seq_len]

        # Repeat keys and values to match query heads (for grouped query attention)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention computation: (B, seq_len, H, Head_Dim) -> (B, H, seq_len, Head_Dim)
        query = query.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Create causal mask for autoregressive attention
        if seq_len > 1:
            mask = torch.full(
                (seq_len, start_pos + seq_len), float("-inf"), device=x.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1)
            mask = mask[None, None, :, :].expand(batch_size, self.n_q_heads, -1, -1)
        else:
            mask = None

        # Compute attention
        output = self.attention(query, keys, values, self.head_dim, mask)

        # Reshape output: (B, H, seq_len, Head_Dim) -> (B, seq_len, H * Head_Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Apply output projection
        return self.wo(output)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = MultiHeadAttentionBlock(args)
        self.feed_forward = FeedForward(args)

        # Normalization before attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # Attention with residual connection
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # Feed forward with residual connection
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        # Final normalization
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Output projection
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # Precompute rotary embedding frequencies
        self.register_buffer(
            "freqs_complex",
            precompute_theta_pos_frequencies(
                args.dim // args.n_heads, args.max_seq_len, device=args.device
            ),
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape

        # Convert tokens to embeddings
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Apply all encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_complex)

        # Final normalization
        h = self.norm(h)

        # Output projection
        output = self.output(h).float()
        return output
