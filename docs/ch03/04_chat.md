# チャットインターフェース

## 学習目標

SFTで学習したモデルを使った**対話型インターフェース**の実装を理解する。

## 主要概念

### 1. プロンプトのフォーマット

```python
def format_prompt(user_message):
    return f"### Instruction:\n{user_message}\n\n### Response:\n"
```

学習時と同じAlpaca形式を使用することが重要。

### 2. チャットループ

```python
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    # プロンプトのフォーマットと生成
    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    # アシスタントの応答部分のみ抽出
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # 出力
    if "\n" in response:
        print(f"Bot:\n{response}")
    else:
        print(f"Bot: {response}")
```

### 3. 応答の抽出

生成されたテキストには入力プロンプトも含まれるため、応答部分のみを抽出：

```python
# 生成結果: "### Instruction:\nHello\n\n### Response:\nHi there!"
response = response.split("### Response:")[-1].strip()
# → "Hi there!"
```

## 使用例

```
You: Write a function to add two numbers

Bot:
def add(a, b):
    return a + b

You: What is Python?

Bot: Python is a programming language.
```

## 設定パラメータ

```python
model_path = 'codebot/model_sft.pt'
tokenizer_path = 'codebot/merge_rules.pkl'
max_new_tokens = 200
temperature = 1.0
```

## ポイント

1. **フォーマットの一貫性**: 学習時と同じ形式を使用
2. **応答の後処理**: プロンプト部分を除去
3. **改行の処理**: 複数行の応答を適切に表示
4. **temperature**: 1.0で自然な応答、0で決定的な応答
