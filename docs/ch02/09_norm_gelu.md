# LayerNorm・GELU・FFN

## 学習目標

Transformerブロックの構成要素である**LayerNorm**、**GELU**、**FFN**を理解する。

## 主要概念

### 1. Layer Normalization

各サンプルの特徴量を正規化する：

```python
class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta
```

- **gamma**: スケールパラメータ（学習可能）
- **beta**: シフトパラメータ（学習可能）
- **eps**: ゼロ除算防止

### 2. GELU活性化関数

Gaussian Error Linear Unit：

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

- ReLUより滑らかな活性化
- 負の値を完全に0にしない
- GPT-2で採用

### 3. Feed-Forward Network (FFN)

```python
class FFN(nn.Module):
    def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)

        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)
```

- 隠れ層は通常4倍の次元
- 位置ごとに独立に処理

### 4. Transformerブロック

```python
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Pre-Norm + 残差接続
        x = x + self.ffn(self.norm2(x))   # Pre-Norm + 残差接続
        return x
```

## Pre-Norm vs Post-Norm

| 方式 | 構造 | 特徴 |
|------|------|------|
| Post-Norm | x + Norm(Attn(x)) | 元のTransformer |
| **Pre-Norm** | x + Attn(Norm(x)) | GPT-2採用、学習が安定 |

## ポイント

1. **残差接続**: 勾配消失を防ぎ、深いネットワークの学習を可能に
2. **Pre-Norm**: 正規化を先に適用することで学習が安定
3. **FFNの役割**: 各位置で非線形変換を行い表現力を向上
