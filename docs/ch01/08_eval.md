# BPEトークナイザーの評価

## 学習目標

学習済みBPEトークナイザーの**品質を評価**する方法を学ぶ。トークンの内容確認と圧縮率の測定。

## 主要概念

### 1. トークナイザーの読み込み

```python
from codebot.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
```

### 2. 学習されたトークンの確認

```python
# 最初に学習された10個
print("最初に学習された10個:")
for token_id in range(256, 266):
    byte_seq = tokenizer.id_to_byte[token_id]
    text = byte_seq.decode('utf-8', errors='replace')
    print(f"  ID {token_id}: '{text}'")

# 最後に学習された10個
print("\n最後に学習された10個:")
for token_id in range(990, 1000):
    byte_seq = tokenizer.id_to_byte[token_id]
    text = byte_seq.decode('utf-8', errors='replace')
    print(f"  ID {token_id}: '{text}'")
```

### 3. 学習順序の意味

| 学習順序 | 特徴 |
|----------|------|
| 最初（ID 256〜） | 最も頻出するペア。短い文字列が多い |
| 最後（〜vocab_size） | 頻度が低いペア。長い文字列になる傾向 |

### 4. 圧縮率の測定

```python
sample_text = open("codebot/tiny_codes.txt").read()[:10000]

byte_count = len(sample_text.encode('utf-8'))
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print(f"=== 圧縮効率 ===")
print(f"バイト数: {byte_count:,}")
print(f"トークン数: {ids_count:,}")
print(f"圧縮率: {compression_ratio:.2f}倍")
```

### 5. 圧縮率の解釈

```
圧縮率 = バイト数 / トークン数
       = 平均バイト数/トークン
```

| 圧縮率 | 意味 |
|--------|------|
| 1.0 | バイトトークナイザーと同等 |
| 2.0 | 1トークンあたり平均2バイト |
| 4.0 | 効率的な圧縮 |

### 6. 他のトークナイザーとの比較（参考）

```python
import tiktoken

text = open("codebot/tiny_codes.txt").read()[:10000]
byte_count = len(text.encode('utf-8'))

for name, encoding_name in [('GPT-2', 'gpt2'), ('cl100k_base', 'cl100k_base')]:
    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text, allowed_special={'<|endoftext|>'}))
    ratio = byte_count / token_count
    print(f"{name}: 語彙数 {encoding.n_vocab:,}, 圧縮率 {ratio:.2f}倍")
```

## 評価のポイント

### 良いトークナイザーの特徴

1. **高い圧縮率**: 少ないトークンでテキストを表現
2. **意味的なまとまり**: 単語やサブワードが適切に分割
3. **頻出パターンの捕捉**: よく使われるフレーズがトークン化

### コードデータ向けトークナイザーの期待

- インデント（スペース）のまとまり
- キーワード（`def`, `return`, `if` など）
- 変数名のパターン

## 実行方法

```bash
cd ch01
python 08_eval.py
```

## 出力例

```
最初に学習された10個:
  ID 256: '  '     # 2スペース
  ID 257: 'in'
  ID 258: 'the'
  ...

圧縮率: 3.45倍（平均 3.45 バイト/トークン）
```

## 次のステップ

圧縮率が十分であれば、このトークナイザーを使って：
- 事前学習用データのエンコード（09_bpe_encode.py）
- モデルの学習（ch03）
