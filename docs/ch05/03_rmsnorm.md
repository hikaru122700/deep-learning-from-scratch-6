# RMSNorm

## 学習目標

LayerNormの簡略版である**RMSNorm**を理解する。LLaMAなど多くの最新モデルで採用。

## 主要概念

### 1. LayerNormの復習

```python
# LayerNorm
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
norm_x = (x - mean) / sqrt(var + eps)
output = gamma * norm_x + beta
```

- 平均を引く（センタリング）
- 分散で割る（スケーリング）
- gamma, beta でアフィン変換

### 2. RMSNormの簡略化

**平均を引く操作を省略**：

```python
class RMSNorm(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(x))
        self.eps = 1e-5

    def forward(self, x):
        x2 = x**2
        ms = x2.mean(dim=-1, keepdim=True)  # Mean Square
        rms = torch.sqrt(ms + self.eps)      # Root Mean Square
        return self.gamma * x / rms
```

### 3. 数式

```
RMSNorm(x) = gamma * x / sqrt(mean(x²) + eps)
```

### 4. LayerNormとの比較

| 項目 | LayerNorm | RMSNorm |
|------|-----------|---------|
| センタリング | あり（x - mean） | なし |
| スケーリング | あり | あり |
| パラメータ | gamma, beta | gamma のみ |
| 計算量 | やや多い | 少ない |

## なぜRMSNormが機能するか

研究により、LayerNormの効果の大部分は**スケーリング**によるものと判明：
- センタリングは性能にあまり寄与しない
- 計算を省略しても精度はほぼ同等

## 使用例

```python
# Blockクラスでの使用
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, rope=None):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)  # PyTorch組み込み
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, rope)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, ff_dim)
```

## ポイント

1. **パラメータ削減**: betaが不要
2. **計算効率**: 平均計算が不要
3. **PyTorch対応**: `nn.RMSNorm` として組み込み
