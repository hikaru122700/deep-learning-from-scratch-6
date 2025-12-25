# Attention出力変換（Output Projection）

## 学習目標

Attentionの出力を**出力変換行列 W_o** で変換する仕組みを理解する。

## 主要概念

### 1. なぜ出力変換が必要か

- Attention後の hidden は head_dim 次元
- 元の embed_dim に戻す必要がある
- さらに表現力を高める

### 2. 拡張されたAttentionクラス

```python
class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, key_dim, bias=False)  # key_dimに変更
        self.W_o = nn.Linear(key_dim, embed_dim, bias=False)  # 出力変換行列
        self.key_dim = key_dim

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        K_t = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_t)
        scores = scores / (self.key_dim ** 0.5)

        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)

        # 出力変換
        output = self.W_o(hidden)

        return output
```

### 3. 次元の流れ

```
入力: x (B, C, embed_dim)
      ↓ W_q, W_k, W_v
Q, K, V (B, C, key_dim)
      ↓ Attention計算
hidden (B, C, key_dim)
      ↓ W_o
出力 (B, C, embed_dim)
```

### 4. 使用例

```python
attention = Attention(embed_dim=256, key_dim=64)
x = torch.randn(2, 5, 256)  # (batch=2, seq=5, embed=256)
y = attention(x)

print("入力形状:", x.shape)   # (2, 5, 256)
print("出力形状:", y.shape)   # (2, 5, 256)
```

## ポイント

1. **W_o の役割**: key_dim → embed_dim の変換
2. **残差接続との互換性**: 入出力の次元が一致する必要がある
3. **表現力の向上**: W_o も学習パラメータとして最適化される

## パラメータ数

| 行列 | サイズ | パラメータ数 |
|------|--------|-------------|
| W_q | (embed_dim, key_dim) | E × D |
| W_k | (embed_dim, key_dim) | E × D |
| W_v | (embed_dim, key_dim) | E × D |
| W_o | (key_dim, embed_dim) | D × E |
| **合計** | | 4 × E × D |
