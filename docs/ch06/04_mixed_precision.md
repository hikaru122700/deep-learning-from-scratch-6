# 混合精度学習（Mixed Precision Training）

## 学習目標

**FP16/BF16**を活用した効率的な学習方法を理解する。

## 主要概念

### 1. 浮動小数点精度の種類

| 精度 | ビット数 | 範囲 | 用途 |
|------|---------|------|------|
| FP32 | 32 | 広い | 標準 |
| FP16 | 16 | 狭い | 高速だが不安定 |
| BF16 | 16 | 広い | 高速で安定 |

### 2. FP16の問題点

```python
# 精度の損失
large = torch.tensor(1000.0, dtype=torch.float16)
small = torch.tensor(0.01, dtype=torch.float16)
print(large + small)  # tensor(1000.)  ← 0.01が消える

# アンダーフロー
tiny = torch.tensor(1e-8, dtype=torch.float16)
print(tiny)  # tensor(0.)

# オーバーフロー
huge = torch.tensor(70000.0, dtype=torch.float16)
print(huge)  # tensor(inf)
```

### 3. BF16の利点

```python
# BF16ではアンダーフローしない
tiny_bf16 = torch.tensor(1e-8, dtype=torch.bfloat16)
print(tiny_bf16)  # tensor(1.0012e-08)

# BF16ではオーバーフローしない
huge_bf16 = torch.tensor(70000.0, dtype=torch.bfloat16)
print(huge_bf16)  # tensor(70144.)  ← 精度は落ちるが表現可能
```

### 4. 自動混合精度（AMP）

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    b = a @ a   # 行列積はBF16
    c = a.sum() # 累積はFP32（自動的に判断）
    print(b.dtype)  # torch.bfloat16
    print(c.dtype)  # torch.float32
```

### 5. 学習ループでの使用

```python
for i in pbar:
    batch_x, batch_y = get_batch(...)
    optimizer.zero_grad()

    # 混合精度で順伝播と損失計算
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    loss.backward()
    optimizer.step()
```

## 演算ごとの精度選択（自動）

| 演算 | 使用精度 | 理由 |
|------|---------|------|
| 行列積 | BF16 | 高速化のメリット大 |
| 累積（sum） | FP32 | 精度が必要 |
| Softmax | FP32 | 数値安定性 |
| LayerNorm | FP32 | 統計計算の精度 |

## ポイント

1. **BF16推奨**: FP16より数値的に安定
2. **autocast**: 自動で最適な精度を選択
3. **GPUサポート**: A100以降はBF16を効率的に処理
4. **CPUでも動作**: 速度向上は限定的
