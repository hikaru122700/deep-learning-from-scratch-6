# 事前学習（Pre-training）

## 学習目標

GPTモデルの**事前学習**（言語モデリング）の実装を理解する。

## 主要概念

### 1. データセットクラス

```python
class TokenDataset(Dataset):
    def __init__(self, tokens, context_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x, y
```

- **x**: 入力トークン（位置 0〜C-1）
- **y**: 正解トークン（位置 1〜C）← 1つずれている

### 2. データ準備

```python
ids = np.fromfile(data_path, dtype=np.uint16)
dataset = TokenDataset(ids, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### 3. 学習ループ

```python
data_iter = cycle(dataloader)  # 無限ループ化
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (B*C, vocab_size)
        batch_y.view(-1)                    # (B*C,)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
```

### 4. 損失関数

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

- **Cross Entropy**: 予測確率分布と正解の差を測定
- 各位置での次トークン予測の誤差を平均

### 5. ハイパーパラメータ

```python
context_len = 256
vocab_size = 1000
batch_size = 32
learning_rate = 3e-4
max_iters = 20000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout = 0.1
```

## 学習の流れ

```
1. バイナリファイルからトークンIDを読み込み
2. DataLoaderでバッチを作成
3. モデルで次トークンを予測
4. Cross Entropy Lossを計算
5. 逆伝播で勾配を計算
6. AdamWで重みを更新
7. 繰り返し
```

## ポイント

1. **cycle(dataloader)**: データセットを何周でも使える
2. **view(-1, ...)**: バッチとシーケンスを1次元に展開
3. **tqdm**: 進捗バーを表示
