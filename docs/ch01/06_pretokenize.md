# 事前トークン化（Pre-tokenization）

## 学習目標

BPE学習・エンコード前に行う**事前トークン化（Pre-tokenization）** の仕組みと重要性を理解する。

## 主要概念

### 1. 事前トークン化とは

テキストを単語や記号などの小さな単位に分割してから、BPEを適用する手法。

**なぜ必要か？**
- 単語を跨いだマージを防ぐ
- 意味的に不自然なトークンの生成を防ぐ
- GPT-2やGPT-4で採用されている

### 2. GPT-2スタイルの正規表現パターン

```python
import regex as re

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

| パターン | マッチする例 |
|----------|-------------|
| `'(?:[sdmt]\|ll\|ve\|re)` | 's, 'd, 'm, 't, 'll, 've, 're（縮約形） |
| `?\p{L}+` | 英単語、日本語など（先頭スペース含む） |
| `?\p{N}+` | 数字（先頭スペース含む） |
| `?[^\s\p{L}\p{N}]+` | 記号類（先頭スペース含む） |
| `\s+(?!\S)` | 末尾の空白 |
| `\s+` | その他の空白 |

### 3. 事前トークン化の実装

```python
def pretokenize_iter(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)
```

ジェネレータを使用してメモリ効率を向上。

### 4. 学習への組み込み

```python
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    texts = input_text.split(end_token)

    # 各テキスト片を事前トークン化
    ids_list = []
    for text in texts:
        for pretoken in pretokenize_iter(text):
            ids_list.append(list(pretoken.encode("utf-8")))

    # 以降は通常のBPE学習...
```

### 5. エンコードへの組み込み

```python
def encode(self, input_text, show_progress=False):
    pattern = '(' + re.escape(self.end_token) + ')'
    texts = re.split(pattern, input_text)
    all_ids = []

    for text in texts:
        if text == self.end_token:
            all_ids.append(self.end_token_id)
        else:
            # 各事前トークンをBPEエンコード
            for pretoken in pretokenize_iter(text):
                ids = self._encode_text(pretoken)
                all_ids.extend(ids)

    return all_ids
```

### 6. 使用例

```python
sample_text = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"
merge_rules = train_bpe(sample_text, vocab_size=270)
tokenizer = BPETokenizer(merge_rules)

text = "Say hello!"
ids = tokenizer.encode(text)

# 各トークンIDをデコードして確認
for token_id in ids:
    print(f"{token_id} -> '{tokenizer.decode([token_id])}'")
```

## 事前トークン化の効果

### Before（事前トークン化なし）

```
"hello world" → マージにより "hello w" のようなトークンが生成される可能性
```

### After（事前トークン化あり）

```
"hello world" → [" hello", " world"] → 単語単位でのみマージ
```

## ポイント

1. **regexライブラリ**: 標準のreモジュールではなく `regex` を使用（`\p{L}` などのUnicodeプロパティ対応）
2. **先頭スペース保持**: `" hello"` のようにスペースを含めることで、単語境界を維持
3. **tqdmで進捗表示**: 大規模データの処理時に進捗を可視化

## 依存ライブラリ

```bash
pip install regex tqdm
```
