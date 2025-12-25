# Alpaca形式（Instruction Format）

## 学習目標

SFT（教師あり微調整）のための**Alpaca形式**データフォーマットを理解する。

## 主要概念

### 1. Alpaca形式とは

指示（Instruction）と応答（Response）を明確に区切ったフォーマット：

```
### Instruction:
{instruction}

### Response:
{response}<|endoftext|>
```

### 2. JSONデータの構造

```python
# tiny_codes_sft.json
[
    {
        "instruction": "Hello",
        "response": "Hello. What can I help you with?"
    },
    ...
]
```

### 3. テキストへの変換

```python
item = data[0]
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"

# 結果:
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>
```

### 4. トークン化

```python
token_ids = tokenizer.encode(text)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, ...]
```

## なぜAlpaca形式を使うか

1. **構造化**: モデルが指示と応答を区別できる
2. **一貫性**: 学習と推論で同じフォーマット
3. **汎用性**: 様々なタスクに適用可能

## 他のフォーマット例

| フォーマット | 特徴 |
|-------------|------|
| Alpaca | シンプル、広く使用 |
| ChatML | OpenAI形式、ロール指定 |
| Llama | システムプロンプト対応 |

## ポイント

1. **終了トークン**: `<|endoftext|>` で応答の終わりを示す
2. **改行の扱い**: `\n\n` で視覚的に区切る
3. **トークン化後**: 特殊トークンも含めて学習
