# BPEトークナイザー（BPE Tokenizer）

## 学習目標

学習済みのマージルールを使って、テキストをエンコード/デコードする**BPETokenizer**クラスを実装する。

## 主要概念

### 1. BPETokenizerの構造

```python
class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        # IDからバイト列への対応表（0~255を登録）
        self.id_to_byte = {i: bytes([i]) for i in range(256)}

        # マージされたトークンは元のトークンのバイト列を連結
        for (token1, token2), new_id in merge_rules.items():
            self.id_to_byte[new_id] = self.id_to_byte[token1] + self.id_to_byte[token2]

        self.vocab_size = len(self.id_to_byte)
```

### 2. id_to_byte辞書の構築

マージルールから、各トークンIDが表すバイト列を構築。

```
ID 0-255:   単一バイト（bytes([i])）
ID 256:     token1のバイト列 + token2のバイト列
ID 257:     ...
```

例: `(105, 115) → 256` の場合
- `id_to_byte[256] = bytes([105]) + bytes([115]) = b'is'`

### 3. エンコード処理

```python
def encode(self, text):
    ids = list(text.encode("utf-8"))

    # 学習時の順序でマージルールを適用
    for merge_pair, new_id in self.merge_rules.items():
        ids = merge(ids, merge_pair, new_id)

    return ids
```

**重要**: マージルールは学習時と同じ順序で適用する必要がある。

### 4. デコード処理

```python
def decode(self, ids):
    # 各トークンIDを対応するバイト列に変換
    byte_list = [self.id_to_byte[i] for i in ids]

    # すべてのバイト列を連結
    combined_bytes = b"".join(byte_list)

    # バイト列をUTF-8テキストに変換
    text = combined_bytes.decode("utf-8", errors="replace")
    return text
```

### 5. 使用例

```python
# 学習済みのマージルール
merge_rules = {(105, 115): 256, (256, 32): 257, (105, 110): 258, (72, 101): 259}

tokenizer = BPETokenizer(merge_rules)

text = "Hello世界😁"
ids = tokenizer.encode(text)
# [259, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]

decoded = tokenizer.decode(ids)
# "Hello世界😁"
```

## エンコードの流れ

```
入力: "Hello"
    ↓ UTF-8エンコード
[72, 101, 108, 108, 111]
    ↓ ルール(72, 101) → 259 を適用
[259, 108, 108, 111]
    ↓ 他のルールは該当なし
[259, 108, 108, 111]  # 最終結果
```

## デコードの流れ

```
入力: [259, 108, 108, 111]
    ↓ id_to_byteで変換
[b'He', b'l', b'l', b'o']
    ↓ 連結
b'Hello'
    ↓ UTF-8デコード
"Hello"
```

## ポイント

1. **マージルールの順序保持**: Pythonの辞書は挿入順序を保持（Python 3.7+）
2. **errors="replace"**: 不正なUTF-8バイト列は置換文字（�）に変換
3. **可逆性**: 正しいエンコード→デコードで元のテキストに戻る

## 注意点

- マージルールの適用順序が異なると、異なるトークン列になる可能性がある
- 大きなテキストでは効率が悪い（後の章で最適化）
