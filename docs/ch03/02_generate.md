# テキスト生成（Text Generation）

## 学習目標

学習済みGPTモデルを使って**テキストを生成**する方法を理解する。

## 主要概念

### 1. 生成関数

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()

    # プロンプトをトークン化
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated_ids = ids.clone()

    # トークン生成ループ
    for _ in range(max_new_tokens):
        # コンテキスト長を超えた場合は末尾のみ使用
        if ids.size(1) > model.context_len:
            ids = ids[:, -model.context_len:]

        # 次のトークンを予測
        logits = model(ids)[:, -1, :]  # 最後の位置のlogits

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 終了トークンが生成されたら停止
        if next_id.item() == tokenizer.end_token_id:
            break

        # 生成したトークンを追加
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    return tokenizer.decode(generated_ids[0].tolist())
```

### 2. Temperature（温度）パラメータ

```python
probs = F.softmax(logits / temperature, dim=-1)
```

| Temperature | 効果 |
|-------------|------|
| 0 | 決定的（最高確率のトークンを選択） |
| 0.5 | やや保守的 |
| 1.0 | 通常（学習時と同じ） |
| 1.5+ | よりランダム・創造的 |

### 3. サンプリング

```python
# Greedy（temperature=0）
next_id = logits.argmax(dim=-1, keepdim=True)

# Sampling（temperature>0）
next_id = torch.multinomial(probs, num_samples=1)
```

### 4. 使用例

```python
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

for i in range(5):
    print(f"--- サンプル {i+1} ---")
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="def",
        max_new_tokens=200,
        temperature=1.0
    )
    print(generated_text)
```

## 生成の流れ

```
prompt: "def"
    ↓ encode
ids: [256, 101, 102]
    ↓ model(ids)
logits: (1, 3, vocab_size)
    ↓ 最後の位置を取得
logits[:, -1, :]: (1, vocab_size)
    ↓ softmax(logits / temperature)
probs: (1, vocab_size)
    ↓ multinomial sampling
next_id: 123
    ↓ cat
ids: [256, 101, 102, 123]
    ↓ 繰り返し...
```

## ポイント

1. **@torch.no_grad()**: 勾配計算を無効化（メモリ節約・高速化）
2. **model.eval()**: Dropoutなどを無効化
3. **コンテキスト長の制限**: 長いテキストは末尾のみ使用
