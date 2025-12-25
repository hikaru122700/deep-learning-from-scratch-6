# テキスト生成（StoryBot）

## 学習目標

学習済みStoryBotモデルを使った**物語生成**を理解する。

## 主要概念

### 1. 設定

```python
device = get_device()
model_path = 'storybot/model_pretrain.pt'
tokenizer_path = 'storybot/merge_rules.pkl'

prompt = "<|endoftext|>"  # 新しい物語を開始
max_new_tokens = 1000
temperature = 1.0
num_samples = 3
```

### 2. モデル読み込みと生成

```python
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

for i in range(num_samples):
    print(f"--- サンプル {i+1} ---")
    generated_text = generate(
        model, tokenizer, prompt, max_new_tokens, temperature
    )
    print(generated_text)
```

### 3. プロンプトの選択

| プロンプト | 用途 |
|-----------|------|
| `<\|endoftext\|>` | 新しい物語を開始 |
| `"Once upon a time"` | 特定の書き出しから続ける |
| `"The little girl"` | 特定のキャラクターで開始 |

### 4. generate関数の動作

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature):
    model.eval()

    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if ids.size(1) > model.context_len:
            ids = ids[:, -model.context_len:]

        logits = model(ids)[:, -1, :]

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat((ids, next_id), dim=1)

    return tokenizer.decode(ids[0].tolist())
```

## 生成パラメータの影響

| パラメータ | 低い値 | 高い値 |
|-----------|-------|-------|
| temperature | 決定的・反復的 | ランダム・創造的 |
| max_new_tokens | 短い文 | 長い物語 |

## 出力例

```
--- サンプル 1 ---
Once upon a time, there was a little girl named Lily.
She loved to play in the garden with her friends.
One day, she found a beautiful flower...
```

## ポイント

1. **@torch.no_grad()**: 勾配計算なしで高速生成
2. **model.eval()**: Dropoutを無効化
3. **終了トークン**: `<|endoftext|>` で生成を停止
