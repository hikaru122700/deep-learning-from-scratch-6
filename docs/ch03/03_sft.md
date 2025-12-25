# SFT（教師あり微調整）

## 学習目標

事前学習済みモデルを**Instruction-Following**タスクに微調整する方法を理解する。

## 主要概念

### 1. SFTデータセット

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            ids, labels = self._create_sample(item['instruction'], item['response'])
            self.samples.append((ids, labels))

    def _create_sample(self, instruction, response):
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"

        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        # 入力系列とラベルの作成
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids  # プロンプト部分をマスク

        # 言語モデル用にシフト
        ids = ids[:-1]
        labels = labels[1:]

        # パディング処理...
        return ids, labels
```

### 2. ラベルのマスク

```
プロンプト: "### Instruction:\nHello\n\n### Response:\n"
応答:      "Hi there!<|endoftext|>"

ids:    [35, 35, 35, ..., 72, 105, ...]
labels: [-100, -100, -100, ..., 72, 105, ...]
        ↑ プロンプト部分は-100   ↑ 応答部分は実際のID
```

### 3. 損失計算

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    batch_y.view(-1),
    ignore_index=-100  # -100のラベルは損失計算から除外
)
```

### 4. 学習ループ

```python
model = GPT.load_from(pretrain_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_y.view(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## ハイパーパラメータ

```python
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 500  # 事前学習より少ない
```

## SFTのポイント

1. **事前学習モデルから開始**: `GPT.load_from()` で読み込み
2. **プロンプトをマスク**: 応答部分のみで損失を計算
3. **少ないイテレーション**: 事前学習の知識を活かす
4. **低い学習率**: 微調整なので大きな変更は避ける

## 事前学習 vs SFT

| 項目 | 事前学習 | SFT |
|------|---------|-----|
| 目的 | 言語理解 | 指示に従う |
| データ | 大量のテキスト | 指示-応答ペア |
| イテレーション | 多い（20000+） | 少ない（500程度） |
| 損失計算 | 全トークン | 応答部分のみ |
