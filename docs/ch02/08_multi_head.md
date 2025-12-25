# Multi-Head Attention

## 学習目標

複数のAttention（ヘッド）を並列に計算する**Multi-Head Attention**を理解する。

## 主要概念

### 1. なぜマルチヘッドか

- 単一のAttentionでは1つの視点からしか情報を集約できない
- 複数のヘッドで異なる特徴を捉える
- 例：文法的関係、意味的関係、位置関係など

### 2. 効率的な実装

```python
# 全ヘッド分の重みを一つの行列にまとめる
W_q = nn.Linear(E, H*D, bias=False)  # H: ヘッド数, D: ヘッド次元
W_k = nn.Linear(E, H*D, bias=False)
W_v = nn.Linear(E, H*D, bias=False)

Q = W_q(x)  # (B, C, H*D)
K = W_k(x)
V = W_v(x)

# 形状の変換: (B, C, H*D) → (B, H, C, D)
Q = Q.view(B, C, H, D).transpose(1, 2)
K = K.view(B, C, H, D).transpose(1, 2)
V = V.view(B, C, H, D).transpose(1, 2)
```

### 3. MultiHeadAttentionクラス

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim

        # Q, K, V の計算と形状変換
        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, C, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, C, H, D).transpose(1, 2)

        # Attentionスコア
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)

        # マスク
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 重み
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V)  # (B, H, C, D)

        # ヘッドの結合
        hidden = hidden.transpose(1, 2).contiguous().view(B, C, H*D)
        output = self.W_o(hidden)
        output = self.output_dropout(output)

        return output
```

### 4. テンソル形状の変化

```
入力:     (B, C, E)           例: (2, 10, 512)
    ↓ W_q, W_k, W_v
Q,K,V:    (B, C, H*D)         例: (2, 10, 512)
    ↓ view + transpose
Q,K,V:    (B, H, C, D)        例: (2, 8, 10, 64)
    ↓ Attention
hidden:   (B, H, C, D)        例: (2, 8, 10, 64)
    ↓ transpose + view
hidden:   (B, C, H*D)         例: (2, 10, 512)
    ↓ W_o
出力:     (B, C, E)           例: (2, 10, 512)
```

## ポイント

1. **並列計算**: 全ヘッドを1回の行列演算で処理
2. **Dropout**: Attention重みと出力の両方に適用
3. **contiguous()**: メモリレイアウトを連続化（view前に必要）

## ハイパーパラメータ例

```python
embed_dim = 512
n_head = 8
head_dim = 64  # = embed_dim / n_head
```
