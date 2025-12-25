# 文字トークナイザー（Character Tokenizer）

## 学習目標

このファイルでは、最もシンプルなトークナイザーである**文字レベルトークナイザー**の仕組みを学ぶ。

## 主要概念

### 1. 文字列のリスト化

Pythonでは文字列を `list()` で分割すると、1文字ずつのリストになる。

```python
text = "hello世界😁"
print(list(text))  # ['h', 'e', 'l', 'l', 'o', '世', '界', '😁']
```

- 英字、日本語、絵文字など、すべて1文字として扱われる
- Pythonの文字列はUnicodeベース

### 2. 文字とコードポイントの変換

| 関数 | 役割 | 例 |
|------|------|-----|
| `ord()` | 文字 → Unicodeコードポイント（整数） | `ord('h')` → `104` |
| `chr()` | Unicodeコードポイント → 文字 | `chr(104)` → `'h'` |

```python
print(ord('😁'))   # 128513
print(chr(128513)) # '😁'
```

### 3. CharTokenizerクラス

```python
class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ''.join([chr(i) for i in ids])
```

#### メソッド

- **encode**: テキスト → IDリスト（整数のリスト）
- **decode**: IDリスト → テキスト

### 4. 使用例

```python
tokenizer = CharTokenizer()
text = "hello世界😁"

ids = tokenizer.encode(text)
# [104, 101, 108, 108, 111, 19990, 30028, 128513]

decoded = tokenizer.decode(ids)
# "hello世界😁"
```

## ポイント

1. **可逆性**: encode → decode で元のテキストに戻せる
2. **語彙サイズ**: Unicodeの全コードポイント数（約14万文字）が語彙サイズとなる
3. **シンプルさ**: 学習不要で即座に使える

## 課題・限界

- 語彙サイズが非常に大きい
- 1文字=1トークンなので、シーケンス長が長くなる
- 単語やサブワードの意味的なまとまりを考慮しない

→ これらの課題を解決するため、次のステップでBPE（Byte Pair Encoding）などを学ぶ
