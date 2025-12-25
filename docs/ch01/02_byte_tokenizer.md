# バイトトークナイザー（Byte Tokenizer）

## 学習目標

文字単位ではなく**バイト単位**でトークン化する方法を学ぶ。これはBPEの基盤となる重要な概念。

## 主要概念

### 1. UTF-8エンコーディング

文字をバイト列に変換する標準的な方法。

```python
# ASCII文字（1バイト）
encoded = 'A'.encode('utf-8')
print(list(encoded))  # [65]

# 日本語（3バイト）
encoded = 'あ'.encode('utf-8')
print(list(encoded))  # [227, 129, 130]
```

| 文字種 | バイト数 | 例 |
|--------|---------|-----|
| ASCII（英数字） | 1バイト | 'A' → [65] |
| 日本語・中国語 | 3バイト | 'あ' → [227, 129, 130] |
| 絵文字 | 4バイト | '😁' → [240, 159, 152, 129] |

### 2. バイト列からテキストへの復元

```python
ids = [65]
decoded = bytes(ids).decode('utf-8')
print(decoded)  # 'A'
```

### 3. ByteTokenizerクラス

```python
class ByteTokenizer:
    def encode(self, text):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8')
```

### 4. 使用例

```python
tokenizer = ByteTokenizer()
text = "hello世界😁"

ids = tokenizer.encode(text)
# [104, 101, 108, 108, 111, 228, 184, 150, 231, 149, 140, 240, 159, 152, 129]

decoded = tokenizer.decode(ids)
# "hello世界😁"
```

## CharTokenizerとの比較

| 項目 | CharTokenizer | ByteTokenizer |
|------|--------------|---------------|
| 語彙サイズ | 約14万（Unicode全体） | **256**（0-255） |
| "hello世界😁"のトークン数 | 8 | 15 |
| 特徴 | 語彙が大きい | 語彙が小さく固定 |

## ポイント

1. **固定語彙サイズ**: 常に256トークン（0-255のバイト値）
2. **言語非依存**: どんな言語でも同じ方法で処理可能
3. **BPEの基盤**: この256トークンを初期語彙として、BPEでマージしていく

## 課題

- 日本語や絵文字は複数バイトになるため、シーケンスが長くなる
- 1バイト単位では意味的なまとまりがない

→ BPE（Byte Pair Encoding）で頻出するバイトペアをマージして解決
