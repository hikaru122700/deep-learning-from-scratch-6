# RoPE（Rotary Position Embedding）

## 学習目標

**回転位置埋め込み（RoPE）** の仕組みと実装を理解する。絶対位置埋め込みの代替として多くの最新モデルで採用。

## 主要概念

### 1. RoPEの基本アイデア

位置情報を**ベクトルの回転**として表現：
- 位置 m のベクトルは角度 m × θ で回転
- 相対位置は回転角の差として表現される

### 2. 実装

```python
class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim % 2 == 0
        half = key_dim // 2

        # 周波数の計算
        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ((2.0 * half_ids) / key_dim))

        # 位置ごとの角度を事前計算
        positions = torch.arange(max_context_len)
        angles = positions[:, None] * inv_freq[None, :]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, x):
        batch_size, num_head, context_len, key_dim = x.shape

        cos = self.cos_cache[:context_len]
        sin = self.sin_cache[:context_len]

        # 偶数・奇数インデックスに分割
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 回転を適用
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # 元の形状に戻す
        out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        out = out.reshape(batch_size, num_head, context_len, key_dim)
        return out
```

### 3. 回転の数学

2次元ベクトル (x, y) を角度 θ で回転：

```
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```

RoPEでは key_dim/2 個の2次元ペアに対して異なる周波数で回転を適用。

### 4. Multi-Head Attentionへの統合

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, rope=None):
        # ...
        self.rope = rope

    def forward(self, x):
        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, C, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, C, H, D).transpose(1, 2)

        # RoPEの適用（QとKのみ）
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        # 以降は通常のAttention...
```

## RoPEの利点

| 項目 | 絶対位置埋め込み | RoPE |
|------|---------------|------|
| 相対位置 | 暗黙的 | 明示的 |
| 外挿性能 | 低い | 高い |
| パラメータ | 学習可能 | 固定（cos/sin） |

## ハイパーパラメータ

```python
theta = 10000  # 基底周波数
head_dim = 64  # ヘッド次元（偶数）
max_context_len = 1024
```

## ポイント

1. **register_buffer**: 学習されないが保存される
2. **偶奇分割**: 2次元回転のペアを作成
3. **事前計算**: cos/sin はforwardで計算不要
