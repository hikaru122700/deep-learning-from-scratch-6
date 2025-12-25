# Attentionマスク（Causal Mask）

## 学習目標

言語モデルで必要な**因果的マスク**（Causal Mask）を理解する。未来の情報を参照できないようにする仕組み。

## 主要概念

### 1. なぜマスクが必要か

言語モデルは次のトークンを予測するタスク：
- 位置 i のトークンは、位置 0〜i-1 のトークンのみ参照可能
- 位置 i+1 以降（未来）を見てはいけない

### 2. Attentionクラスの実装

```python
class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_dim = key_dim

    def forward(self, x):  # x: (B, C, E)
        Q = self.W_q(x)    # (B, C, D)
        K = self.W_k(x)    # (B, C, D)
        V = self.W_v(x)    # (B, C, E)

        # Attentionスコア
        K_t = K.transpose(-2, -1)  # (B, D, C)
        scores = torch.matmul(Q, K_t)  # (B, C, C)
        scores = scores / (self.key_dim ** 0.5)

        # マスクの適用
        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)  # (B, C, E)
        return output
```

### 3. マスクの仕組み

```python
# 下三角行列を作成
mask = torch.tril(torch.ones(C, C))
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]

# 0の位置を-infに
scores = scores.masked_fill(mask == 0, float('-inf'))

# softmax後、-infは0になる
weights = F.softmax(scores, dim=-1)
```

### 4. マスク適用後の重み

```
位置0: [1.0, 0.0, 0.0, 0.0, 0.0]  ← 自分のみ
位置1: [0.3, 0.7, 0.0, 0.0, 0.0]  ← 0と1を参照
位置2: [0.2, 0.3, 0.5, 0.0, 0.0]  ← 0,1,2を参照
...
```

## ポイント

1. **torch.tril**: 下三角行列を作成
2. **masked_fill**: 条件に合う位置を指定値で置換
3. **-inf → 0**: ソフトマックス後、未来の位置の重みは0になる

## 学習可能なパラメータ

- `W_q`, `W_k`, `W_v`: 入力を Q, K, V に変換する重み行列
- これらが学習を通じて最適化される
