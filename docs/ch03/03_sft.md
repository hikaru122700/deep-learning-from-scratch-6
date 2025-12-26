# SFT（Supervised Fine-Tuning：教師あり微調整）

## 学習目標

事前学習済みモデルを**Instruction-Following**タスクに微調整する方法を理解する。

## SFTとは

SFT（Supervised Fine-Tuning）は、事前学習済みの言語モデルを特定のタスク（主に指示に従う能力）に適応させる手法。事前学習で獲得した言語知識を保持しながら、ユーザーの指示に適切に応答できるよう調整する。

### なぜSFTが必要か

事前学習モデルは「次のトークンを予測する」能力を持つが、以下の問題がある：

1. **指示への応答が不自然**: 質問に対して質問を続けてしまう
2. **フォーマットの不統一**: 出力形式が予測困難
3. **タスク理解の欠如**: 何を求められているか理解できない

```
# 事前学習モデルの応答例
User: Pythonで素数判定関数を書いて
Model: ください。素数判定関数を書いてください。素数判定関数を...（繰り返し）

# SFT後の応答例
User: Pythonで素数判定関数を書いて
Model: def is_prime(n):
           if n < 2:
               return False
           for i in range(2, int(n**0.5) + 1):
               if n % i == 0:
                   return False
           return True
```

## SFTの全体像

```
┌─────────────────────────────────────────────────────────────┐
│                    SFTパイプライン                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 事前学習済みモデル   2. SFTデータセット                   │
│     ┌─────────┐            ┌─────────────────┐              │
│     │  GPT    │            │ Instruction     │              │
│     │ (知識)  │     +      │ Response pairs  │              │
│     └────┬────┘            └────────┬────────┘              │
│          │                          │                       │
│          └──────────┬───────────────┘                       │
│                     ▼                                       │
│          3. ラベルマスキング + 学習                          │
│             ┌─────────────────────┐                         │
│             │ Instruction: -100  │  ← 損失計算から除外       │
│             │ Response: 実際のID │  ← 損失計算に使用         │
│             └──────────┬──────────┘                         │
│                        ▼                                    │
│          4. 微調整済みモデル                                 │
│             ┌─────────────────────┐                         │
│             │ 指示に従えるGPT    │                          │
│             └─────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 主要概念

### 1. SFTデータセットの構造

SFTでは、指示（Instruction）と応答（Response）のペアを学習データとして使用する。

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

#### データセットの流れ

```
1. JSONから読み込み
   {"instruction": "Hello", "response": "Hi there!"}

2. Alpaca形式に変換
   "### Instruction:\nHello\n\n### Response:\nHi there!<|endoftext|>"

3. トークン化
   prompt_ids:   [35, 35, 35, 962, 519, ...]  (Instruction部分)
   response_ids: [72, 105, 584, ...]          (Response部分)

4. ラベル作成（プロンプトをマスク）
   ids:    [35, 35, 35, 962, 519, ..., 72, 105, 584, ...]
   labels: [-100, -100, -100, ..., 72, 105, 584, ...]

5. シフト（言語モデル用）
   入力:  [35, 35, 35, 962, ..., 72, 105, ...]
   正解:  [35, 35, 962, 519, ..., 105, 584, ...]
```

### 2. ラベルのマスク（Label Masking）

SFTの核心部分。プロンプト（指示）部分は損失計算から除外し、応答部分のみで学習する。

```
入力系列:
┌─────────────────────────────────────┬────────────────────────┐
│     Instruction（指示）             │    Response（応答）     │
│  "### Instruction:\nHello\n\n..."  │   "Hi there!<|eos|>"  │
└─────────────────────────────────────┴────────────────────────┘

ラベル:
┌─────────────────────────────────────┬────────────────────────┐
│        -100（マスク）               │     実際のトークンID    │
│   [-100, -100, -100, ...]          │   [72, 105, 584, ...]  │
└─────────────────────────────────────┴────────────────────────┘
         ↑ 損失計算から除外                   ↑ 損失計算に使用
```

#### なぜマスクが必要か

1. **目的の明確化**: モデルに「応答を生成する」ことを学ばせたい
2. **プロンプトは固定**: 指示部分は推論時に与えられるもの
3. **効率的な学習**: 不要な部分で勾配を計算しない

### 3. 損失計算

PyTorchの`cross_entropy`関数は`ignore_index`パラメータをサポートしており、-100のラベルを自動的に除外する。

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # (batch * seq_len, vocab_size)
    batch_y.view(-1),                   # (batch * seq_len,)
    ignore_index=-100                   # -100のラベルは損失計算から除外
)
```

#### 損失計算の詳細

```python
# 具体例
logits = model(input_ids)  # shape: (batch, seq_len, vocab_size)

# 各位置での損失
for i, (logit, label) in enumerate(zip(logits[0], labels[0])):
    if label == -100:
        # この位置は損失計算に含めない
        continue
    else:
        # クロスエントロピー損失を計算
        loss_i = -log(softmax(logit)[label])
```

### 4. 学習ループ

```python
# 事前学習済みモデルを読み込み（重要！）
model = GPT.load_from(pretrain_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

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

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
```

### 5. パディングとトランケーション

固定長のバッチを作成するため、サンプルを調整する。

```python
def _create_sample(self, instruction, response):
    # ... トークン化とラベル作成 ...

    # context_lenに合わせる
    pad_len = self.context_len - len(ids)

    if pad_len > 0:
        # 短い場合: パディング
        ids = ids + [0] * pad_len      # 0でパディング
        labels = labels + [-100] * pad_len  # パディング部分もマスク
    elif pad_len < 0:
        # 長い場合: 切り詰め
        ids = ids[:self.context_len]
        labels = labels[:self.context_len]

    return ids, labels
```

#### パディングの注意点

```
パディングあり:
ids:    [35, 35, 962, ..., 72, 105, 0, 0, 0, 0]
labels: [-100, -100, ..., 105, 584, -100, -100, -100, -100]
                                    ↑ パディング部分もマスク
```

## ハイパーパラメータ

```python
context_len = 256      # コンテキスト長
batch_size = 32        # バッチサイズ
learning_rate = 3e-4   # 学習率（事前学習と同程度か少し低め）
max_iters = 500        # イテレーション数（事前学習より大幅に少ない）
```

### パラメータ選択の指針

| パラメータ | 推奨値 | 理由 |
|-----------|--------|------|
| 学習率 | 1e-5 〜 3e-4 | 高すぎると事前学習の知識を破壊 |
| イテレーション | 500〜2000 | 過学習を防ぐ |
| バッチサイズ | 16〜64 | メモリと安定性のバランス |

## SFTのポイント

1. **事前学習モデルから開始**: `GPT.load_from()` で読み込み、ゼロから学習しない
2. **プロンプトをマスク**: 応答部分のみで損失を計算
3. **少ないイテレーション**: 事前学習の知識を活かし、過学習を防ぐ
4. **低い学習率**: 微調整なので大きな変更は避ける
5. **終了トークン**: `<|endoftext|>` を含めて学習し、応答の終わりを認識させる

## 事前学習 vs SFT

| 項目 | 事前学習 | SFT |
|------|---------|-----|
| 目的 | 言語理解・知識獲得 | 指示に従う能力 |
| データ | 大量のテキスト | 指示-応答ペア |
| データ量 | 数GB〜TB | 数千〜数万サンプル |
| イテレーション | 多い（20000+） | 少ない（500程度） |
| 損失計算 | 全トークン | 応答部分のみ |
| 初期重み | ランダム | 事前学習済み |
| 学習率 | 比較的高い | 低め |

## よくある問題と対処法

### 1. 過学習（Overfitting）

**症状**: 訓練データには完璧に応答するが、新しい指示には対応できない

**対処法**:
- イテレーション数を減らす
- データ拡張（言い換え、ノイズ追加）
- 早期終了（Early Stopping）

### 2. 壊滅的忘却（Catastrophic Forgetting）

**症状**: SFT後に事前学習で獲得した知識を忘れる

**対処法**:
- 学習率を下げる
- イテレーション数を制限
- 一部のパラメータを固定（LoRAなど）

### 3. フォーマット崩れ

**症状**: 応答が中途半端に終わる、フォーマットが崩れる

**対処法**:
- `<|endoftext|>` を含めて学習
- 学習データのフォーマットを統一

## 実装のチェックリスト

- [ ] 事前学習済みモデルを読み込んでいるか
- [ ] ラベルのマスク（-100）が正しく設定されているか
- [ ] `ignore_index=-100` を損失関数に渡しているか
- [ ] 終了トークンが含まれているか
- [ ] パディング部分もマスクされているか
- [ ] 学習率が適切か（高すぎないか）

## 関連ドキュメント

- [Alpaca形式](./03_alpaca.md): SFTで使用するデータフォーマット
- [チャット推論](./04_chat.md): SFT後のモデルで対話する方法
- [事前学習](./01_pretrain.md): SFTの前段階
- [GRPO](./09_grpo.md): SFT後の強化学習による改善
