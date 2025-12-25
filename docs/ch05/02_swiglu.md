# SwiGLU

## 学習目標

LLaMAで採用されている活性化関数**SwiGLU**を理解する。

## 主要概念

### 1. SiLU（Swish）活性化関数

```python
def silu(x):
    return x * torch.sigmoid(x)
```

- ReLUより滑らかな活性化
- 負の値を完全に0にしない
- 自己ゲーティング機構

### 2. GLU（Gated Linear Unit）

出力を2つの経路に分け、片方をゲートとして使用：

```
出力 = 活性化(Wx) * Vx
```

### 3. SwiGLU実装

```python
class SwiGLU(nn.Module):
    def __init__(self, x_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)  # 約2.67倍

        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, hidden_dim, bias=False)
        self.O = nn.Linear(hidden_dim, x_dim, bias=False)

    def forward(self, x):
        a = self.W(x)
        b = self.V(x)

        gated = F.silu(a) * b  # SiLU + ゲーティング
        out = self.O(gated)
        return out
```

### 4. 次元の流れ

```
入力: (B, C, x_dim)
    ↓ W, V（並列）
a, b: (B, C, hidden_dim)
    ↓ silu(a) * b
gated: (B, C, hidden_dim)
    ↓ O
出力: (B, C, x_dim)
```

## GELUとの比較

| 項目 | GELU (GPT-2) | SwiGLU (LLaMA) |
|------|-------------|----------------|
| 隠れ層サイズ | 4 × embed_dim | 8/3 × embed_dim |
| ゲーティング | なし | あり |
| パラメータ数 | 2つの線形層 | 3つの線形層 |
| 計算量 | 低い | やや高い |

## パラメータ数の計算

```
GELU FFN:
  Linear(E, 4E) + Linear(4E, E) = 8E²

SwiGLU:
  Linear(E, 8E/3) × 2 + Linear(8E/3, E) = 8E²
```

hidden_dim を 8/3 倍にすることで、パラメータ数を同程度に保つ。

## ポイント

1. **bias=False**: LLaMAスタイルではバイアスなし
2. **8/3倍**: パラメータ数を調整するための係数
3. **並列計算**: W(x) と V(x) は独立に計算可能
