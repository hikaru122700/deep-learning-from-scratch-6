# 学習率スケジューラ

## 学習目標

学習率を動的に変化させる**学習率スケジューリング**を理解する。

## 主要概念

### 1. Warmup + Linear Decay

```python
def get_lr(it, max_lr, warmup_iters, max_iters):
    # ウォームアップ：0 -> max_lr
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # アニーリング：max_lr -> 0
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0
```

### 2. 学習率の変化

```
lr
 ↑
max_lr ─────────────────────\
                              \
                               \
                                \
 0 ─────────────────────────────────→ iteration
   |← warmup →|← linear decay →|
```

### 3. 学習ループでの使用

```python
for i in pbar:
    # 学習率を更新
    lr = get_lr(i, learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 通常の学習ステップ...
```

### 4. 典型的な設定

```python
max_lr = 0.001       # 最大学習率
warmup_iters = 200   # ウォームアップステップ数
max_iters = 40000    # 総ステップ数
```

## なぜスケジューリングが必要か

| フェーズ | 学習率 | 理由 |
|---------|-------|------|
| 序盤（Warmup） | 低 → 高 | 不安定な初期状態での大きな更新を防ぐ |
| 中盤 | 高 | 効率的に学習を進める |
| 終盤（Decay） | 高 → 低 | 収束を安定させる |

## 他のスケジューリング手法

| 手法 | 特徴 |
|------|------|
| Linear Decay | シンプル、広く使用 |
| Cosine Annealing | 滑らかな減衰 |
| Step Decay | 特定のステップで段階的に減衰 |
| One Cycle | 1回の上昇と下降 |

## ポイント

1. **Warmupは必須**: 大規模モデルでは特に重要
2. **min_lr**: 0ではなく小さな値にすることも
3. **param_groups**: オプティマイザ内の学習率を直接変更
