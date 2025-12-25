# 特殊トークン（Special Tokens）

## 学習目標

**特殊トークン**（`<|endoftext|>`など）の扱い方を学ぶ。テキストの区切りやモデルへの指示に使用される重要な概念。

## 主要概念

### 1. 特殊トークンとは

通常のテキストには現れない、特別な意味を持つトークン。

| トークン | 用途 |
|----------|------|
| `<|endoftext|>` | テキストの終了を示す |
| `<|pad|>` | パディング（長さ調整） |
| `<|unk|>` | 未知のトークン |
| `<|im_start|>` | メッセージ開始（チャット用） |

### 2. 学習時の特殊トークン処理

特殊トークンでテキストを分割し、**マージが特殊トークンを跨がないように**する。

```python
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # 特殊トークンでテキストを分割
    texts = input_text.split(end_token)
    ids_list = [list(text.encode("utf-8")) for text in texts]

    # 基本語彙（0-255）+ 終了トークン用（1個）を除いた分がマージ回数
    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in range(num_merges):
        # 全てのテキストから隣接ペアの頻度を集計
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)

        best_pair = max(counts, key=counts.get)
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # 全てのテキストでマージを実行
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules
```

### 3. 拡張されたBPETokenizer

```python
class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)  # 最後のIDを特殊トークンに割り当て

        # id_to_byte辞書を構築
        self.id_to_byte = {i: bytes([i]) for i in range(256)}
        for (token1, token2), new_id in merge_rules.items():
            self.id_to_byte[new_id] = self.id_to_byte[token1] + self.id_to_byte[token2]
        self.id_to_byte[self.end_token_id] = self.end_token.encode('utf-8')

        self.vocab_size = len(self.id_to_byte)
```

### 4. エンコード時の特殊トークン処理

```python
def encode(self, input_text):
    pattern = '(' + re.escape(self.end_token) + ')'
    texts = re.split(pattern, input_text)
    all_ids = []

    for text in texts:
        if text == self.end_token:
            all_ids.append(self.end_token_id)  # 特殊トークンIDを追加
        else:
            ids = self._encode_text(text)
            all_ids.extend(ids)

    return all_ids
```

### 5. 使用例

```python
sample_text = "Hello world!<|endoftext|>This is BPE training."
merge_rules = train_bpe(sample_text, vocab_size=260)

tokenizer = BPETokenizer(merge_rules)
text = "Hello world!<|endoftext|>"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)      # [..., 259]（259が<|endoftext|>のID）
print(decoded)  # "Hello world!<|endoftext|>"
```

## 語彙サイズの計算

```
vocab_size = 256 + num_merges + num_special_tokens
           = 256 + (vocab_size - 256 - 1) + 1
```

## ポイント

1. **分割して学習**: 特殊トークンを跨いだマージを防ぐ
2. **専用ID割り当て**: 特殊トークンには固有のIDを割り当て
3. **正規表現で分割**: `re.split()` で特殊トークンを保持しながら分割

## なぜ重要か

- モデルがテキストの境界を認識できる
- 複数の文書を連結して学習する際に必要
- チャットモデルではターン（話者の切り替え）を示す
