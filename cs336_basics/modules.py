import torch
import torch.nn as nn
from einops import einsum


class linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        std = (2 / (num_embeddings + embedding_dim)) ** 0.5
        # std = 1
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # return einsum(token_ids, self.weight, "... num_embeddings,num_embeddings,embedidng_dim -> ... embedding_dim")
        return self.weight[token_ids]


class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = ((x**2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5

        return (x / rms * self.weights).to(in_dtype)
        # return x.to(in_dtype)


class swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, **factory_kwargs))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        for w, out_f, in_f in (
            (self.w1_weight, d_ff, d_model),
            (self.w2_weight, d_model, d_ff),
            (self.w3_weight, d_ff, d_model),
        ):
            std = (2 / (in_f + out_f)) ** 0.5
            nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        # print(self.w1_weight.shape, self.w2_weight.shape, self.w3_weight.shape, in_features.shape)
        silu = self.silu(einsum(in_features, self.w1_weight, "  ... d_model,d_ff d_model ->  ... d_ff"))
        w3x = einsum(in_features, self.w3_weight, "  ... d_model,d_ff d_model ->  ... d_ff")
        siluw3x = silu * w3x
        # siluw3x = silu
        return einsum(siluw3x, self.w2_weight, "... d_ff,d_model d_ff-> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        factory_kwargs = {"device": device}
        self.d_k = d_k
        positions = torch.arange(max_seq_len, **factory_kwargs).unsqueeze(1)
        dims = torch.arange(d_k // 2, **factory_kwargs).unsqueeze(0)
        angles = positions / (theta ** (2 * dims / d_k))
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out1, out2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)


class softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = torch.max(x, dim=self.dim, keepdim=True).values
        tmp = x - x_max
        out = torch.exp(tmp) / torch.sum(torch.exp(tmp), dim=self.dim, keepdim=True)
        return out


class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = einsum(Q, K, "... q_len d_k, ... k_len d_k -> ... q_len k_len") / (d_k**0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = einsum(attention_weights, V, "... q_len k_len,... k_len d_v -> ... q_len d_v")
        return attention_output


class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, device, dtype):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.output_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype

        std = (2 / (d_model + d_model)) ** 0.5
        for w in (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, self.output_proj_weight):
            nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        Q = einsum(in_features, self.q_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        K = einsum(in_features, self.k_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        V = einsum(in_features, self.v_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")

        Q = Q.view(*Q.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(*K.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(*V.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)

        scores = einsum(
            Q,
            K,
            "... num_heads seq_len_q head_dim, ... num_heads seq_len_k head_dim -> ... num_heads  seq_len_q  seq_len_k",
        ) / (self.head_dim**0.5)

        device = in_features.device
        dtype = in_features.dtype
        seq_len = scores.shape[-1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = einsum(attention_weights, V, "... seq_len_q seq_len_k, ... seq_len_k d->... seq_len_q d")
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(*attention_output.shape[:1], attention_output.shape[2], self.d_model)
        )
        output = einsum(attention_output, self.output_proj_weight, "... q_len d_in,d_out d_in -> ... q_len d_out")
        return output


class multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device, dtype):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        dims = torch.arange(self.head_dim // 2, device=device).unsqueeze(0)
        angles = positions / (theta ** (2 * dims / self.head_dim))
        # angles = torch.zeros(max_seq_len, device=device).unsqueeze(1)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.output_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        std = (2 / (d_model + d_model)) ** 0.5
        for w in (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, self.output_proj_weight):
            nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = einsum(in_features, self.q_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        K = einsum(in_features, self.k_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        V = einsum(in_features, self.v_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")

        Q = Q.view(*Q.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(*K.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(*V.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        # Q
        cos = self.cos[token_positions].to(dtype=Q.dtype).unsqueeze(1)
        sin = self.sin[token_positions].to(dtype=Q.dtype).unsqueeze(1)
        x1, x2 = Q[..., 0::2], Q[..., 1::2]
        out1, out2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
        # out1, out2 = x1, x2
        Q = torch.stack([out1, out2], dim=-1).flatten(-2)

        # K
        cos = self.cos[token_positions].to(dtype=Q.dtype).unsqueeze(1)
        sin = self.sin[token_positions].to(dtype=Q.dtype).unsqueeze(1)
        x1, x2 = K[..., 0::2], K[..., 1::2]
        out1, out2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
        # out1, out2 = x1, x2
        K = torch.stack([out1, out2], dim=-1).flatten(-2)
        scores = einsum(
            Q,
            K,
            "... num_heads seq_len_q head_dim, ... num_heads seq_len_k head_dim -> ... num_heads  seq_len_q  seq_len_k",
        ) / (self.head_dim**0.5)

        seq_len = scores.shape[-1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = einsum(attention_weights, V, "... seq_len_q seq_len_k, ... seq_len_k d->... seq_len_q d")
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(*attention_output.shape[:1], attention_output.shape[2], self.d_model)
        )
        output = einsum(attention_output, self.output_proj_weight, "... q_len d_in,d_out d_in -> ... q_len d_out")
        return output


class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device, dtype):
        super().__init__()

        self.ln1 = rmsnorm(d_model, device=device, dtype=dtype)
        self.ln2 = rmsnorm(d_model, device=device, dtype=dtype)
        self.attn = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ffn = swiglu(d_model, d_ff, device, dtype)

    def forward(self, in_features, token_positions):
        block1 = self.attn(self.ln1(in_features), token_positions) + in_features
        block2 = self.ffn(self.ln2(block1)) + block1

        # block1 = self.ln1(self.attn(in_features, token_positions) + in_features)
        # block2 = self.ln2(self.ffn(block1) + block1)
        return block2


class transformer_lm(nn.Module):
    def __init__(
        self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, max_seq_len, rope_theta, device, dtype
    ):
        super().__init__()
        self.embedding = embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                transformer_block(d_model, num_heads, d_ff, max_seq_len, rope_theta, device, dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln = rmsnorm(d_model, device=device, dtype=dtype)
        self.lm = linear(d_model, vocab_size, device, dtype)
        self.context_length = context_length

    def forward(self, in_indices):

        x = self.embedding(in_indices)
        batch, seq_len = in_indices.shape
        token_positions = torch.arange(seq_len, device=in_indices.device).view(1, seq_len).expand(batch, seq_len)
        for block in self.layers:
            x = block(x, token_positions)
        logits = self.lm(self.ln(x))
        # logits = self.lm(x)
        return logits
