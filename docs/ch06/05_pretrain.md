# 本格的な事前学習

## 学習目標

混合精度・学習率スケジューリング・勾配クリッピングを統合した**本格的な事前学習**を理解する。

## 主要概念

### 1. データ読み込み（memmap）

```python
train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
```

メモリマップで大規模ファイルを効率的に扱う。

### 2. バッチ取得関数

```python
def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)
        ix = ix[ix + context_len < len(data)]
        if len(ix) == 0:
            return None, None

    x = torch.stack([torch.from_numpy(data[i:i+context_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)
```

### 3. 評価関数

```python
def evaluate(model, val_data, context_len, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx in range(num_batches):
            x, y = get_batch(val_data, context_len, batch_size, device,
                            random=False, offset=batch_idx * batch_size * context_len)
            if x is None:
                break

            with autocast(device_type=device.type):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      y.view(-1), reduction='sum')

            total_loss += loss.item()
            total_tokens += x.numel()

    model.train()
    return total_loss / total_tokens
```

### 4. 学習ループ

```python
for i in pbar:
    # 学習率を更新
    lr = get_lr(i, learning_rate, warmup_iters, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batch_x, batch_y = get_batch(train_data, context_len, batch_size, device)
    optimizer.zero_grad()

    # 混合精度で順伝播
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    loss.backward()

    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    # 定期的に評価
    if (i % eval_iters) == 0:
        val_loss = evaluate(model, val_data, context_len, batch_size, device)
```

### 5. ハイパーパラメータ

```python
context_len = 256
vocab_size = 10000
batch_size = 32
learning_rate = 0.001
warmup_iters = 200
max_iters = 40000
embed_dim = 512
n_head = 16
n_layer = 4
ff_dim = 1344
theta = 10000
eval_iters = 500
grad_clip = 1.0
```

## 学習のポイント

| 項目 | 手法 | 効果 |
|------|------|------|
| 混合精度 | BF16 autocast | メモリ削減、高速化 |
| 学習率 | Warmup + Decay | 安定した収束 |
| 勾配クリッピング | clip_grad_norm_ | 勾配爆発防止 |
| 評価 | Validation Loss | 過学習の検出 |

## チェックポイント保存

```python
save_iters = [500, 5000]

if i in save_iters:
    save_path = f'storybot/model_iter_{i}.pt'
    model.save(save_path)
```

学習途中のモデルを保存して、後で評価に使用。
