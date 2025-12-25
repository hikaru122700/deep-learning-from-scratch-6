# AdamW オプティマイザ

## 学習目標

最も広く使われているオプティマイザ**AdamW**の仕組みを理解する。

## 主要概念

### 1. SGD（確率的勾配降下法）

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - lr * p.grad.data
```

シンプルだが、学習率の調整が難しい。

### 2. AdamWの実装

```python
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 状態の初期化
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)  # 1次モーメント
                    state['v'] = torch.zeros_like(p.data)  # 2次モーメント

                state['t'] += 1
                t = state['t']

                # モーメントの更新
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                state['m'], state['v'] = m, v

                # バイアス補正
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                lr, eps, wd = group['lr'], group['eps'], group['weight_decay']

                # パラメータ更新（Weight Decay分離）
                p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps) - lr * wd * p.data
```

### 3. 各コンポーネントの役割

| コンポーネント | 役割 |
|--------------|------|
| m（1次モーメント） | 勾配の移動平均（モメンタム） |
| v（2次モーメント） | 勾配の2乗の移動平均（適応的学習率） |
| バイアス補正 | 初期ステップでの偏りを補正 |
| Weight Decay | 正則化（過学習防止） |

### 4. ハイパーパラメータ

```python
lr = 1e-3         # 学習率
betas = (0.9, 0.999)  # モーメントの減衰率
eps = 1e-8        # ゼロ除算防止
weight_decay = 0.01   # 正則化強度
```

## Adam vs AdamW

| 項目 | Adam | AdamW |
|------|------|-------|
| Weight Decay | 勾配に含める | 分離 |
| 正則化効果 | 弱い | 強い |
| 推奨 | - | ○ |

## ポイント

1. **state辞書**: パラメータごとにモーメントを保持
2. **β1, β2**: 通常は (0.9, 0.999) で固定
3. **Weight Decay分離**: AdamWの重要な特徴
