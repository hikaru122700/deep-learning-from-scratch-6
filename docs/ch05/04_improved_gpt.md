# 改良版GPT（LLaMAスタイル）

## 学習目標

RoPE、SwiGLU、RMSNormを統合した**改良版GPT**を理解する。

## 主要概念

### 1. 改良点まとめ

| コンポーネント | GPT-2 | 改良版（LLaMAスタイル） |
|--------------|-------|----------------------|
| 位置埋め込み | 学習可能埋め込み | RoPE |
| 正規化 | LayerNorm | RMSNorm |
| FFN | GELU | SwiGLU |
| バイアス | あり | なし |

### 2. 改良版Blockクラス

```python
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim, rope=None):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, rope)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### 3. 改良版GPTクラス

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, n_head, n_layer, ff_dim, theta=10000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 位置埋め込みは不要（RoPEで代替）

        head_dim = embed_dim // n_head
        rope = RoPE(theta, head_dim, context_len)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, ff_dim, rope)
            for _ in range(n_layer)
        ])

        self.norm = nn.RMSNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, ids):
        x = self.embed(ids)  # 位置埋め込みの加算なし
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.unembed(x)
        return logits
```

### 4. ハイパーパラメータ

```python
vocab_size = 10000
context_len = 256
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = int(embed_dim * 8 / 3)  # SwiGLU用
theta = 10000
```

## アーキテクチャ図

```
入力ID: (B, C)
    ↓ Token Embedding（位置埋め込みなし）
埋め込み: (B, C, E)
    ↓ Block × n_layer
        ├── RMSNorm → MHA(+RoPE) → 残差接続
        └── RMSNorm → SwiGLU → 残差接続
    ↓ RMSNorm
    ↓ Unembed
出力: (B, C, vocab_size)
```

## 性能向上の要因

1. **RoPE**: より良い位置情報の表現、長文への汎化
2. **SwiGLU**: より表現力の高い非線形変換
3. **RMSNorm**: 計算効率の向上
4. **バイアスなし**: 過学習の軽減

## ポイント

1. **RoPEの共有**: 全Blockで同じRoPEインスタンスを使用
2. **重み共有なし**: embed と unembed は別々（LLaMAスタイル）
3. **ff_dim**: SwiGLUに合わせて 8/3 倍に設定
